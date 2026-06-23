"""LLM as a Judge：对生成的测试用例做 5 维评分，并驱动低分重生成。

评分维度（各 1-5 分，满分 25）：
- completeness  完整性（前置/步骤/预期是否齐全）
- clarity       清晰度（步骤是否可直接执行）
- correctness   正确性（是否与需求一致）
- coverage      覆盖度（是否符合指定 case_type）
- priority      优先级合理性

合格线 PASS_THRESHOLD 默认 18，可调。
"""
import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.llm import get_generation_llm, get_judge_llm

PASS_THRESHOLD = 18
MAX_RETRY = 2

DIMENSIONS = ["completeness", "clarity", "correctness", "coverage", "priority"]

_JUDGE_PROMPT = """你是一名严格的测试用例评审专家，请对给定的一条测试用例进行 5 维打分。

评分维度（每项 1-5 分）：
- completeness：title / preconditions / steps / expected 是否齐全且内容有效
- clarity：步骤是否清晰、可直接执行
- correctness：用例是否与给定的『功能点』一致
- coverage：用例是否体现了指定 case_type 的测试意图
  （functional=正向功能，boundary=边界值，exception=异常输入/系统异常，negative=非法/反向操作）
- priority：priority 字段设置是否合理（核心主流程应 P0/P1，边界/异常视影响而定）

只输出如下严格 JSON，不要输出 Markdown、不要多余文字：
{"completeness": n, "clarity": n, "correctness": n,
 "coverage": n, "priority": n, "comment": "简短中文点评，指出问题和改进方向"}
"""

_REGEN_PROMPT = """你是资深测试用例设计工程师。上一版用例评分不达标，请根据评审意见重写。

【功能点】{point}
【用例类型】{case_type}
【上一版用例】
{last_case}
【评审反馈】
{feedback}

请输出严格 JSON，字段：title / preconditions / steps(list) / expected / priority(P0-P3)。
不要输出 Markdown 代码块，不要多余文字。
"""

_INIT_PROMPT = """你是资深测试用例设计工程师。请针对给定功能点产出一条 {case_type} 类型的测试用例。

输出严格 JSON：
{{"title":"...", "preconditions":"...", "steps":["..."], "expected":"...", "priority":"P0|P1|P2|P3"}}
不要输出 Markdown，不要多余文字。"""


def _extract_json(text: str) -> dict:
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


def judge_case(point: str, case_type: str, case: dict) -> dict:
    """给一条用例打分。返回 {total, dims, comment, passed}"""
    llm = get_judge_llm()
    user = (
        f"【功能点】{point}\n"
        f"【用例类型】{case_type}\n"
        f"【用例内容】\n{json.dumps(case, ensure_ascii=False, indent=2)}"
    )
    resp = llm.invoke([SystemMessage(_JUDGE_PROMPT), HumanMessage(user)])
    raw = resp.content if isinstance(resp.content, str) else str(resp.content)
    try:
        data = _extract_json(raw)
    except Exception:
        return {
            "total": 0,
            "dims": {d: 0 for d in DIMENSIONS},
            "comment": f"judge 解析失败：{raw[:200]}",
            "passed": False,
        }
    dims = {d: int(data.get(d, 0) or 0) for d in DIMENSIONS}
    total = sum(dims.values())
    return {
        "total": total,
        "dims": dims,
        "comment": str(data.get("comment", "")).strip(),
        "passed": total >= PASS_THRESHOLD,
    }


def generate_with_judge(
    point: str,
    case_type: str,
    max_retry: int = MAX_RETRY,
    threshold: int = PASS_THRESHOLD,
) -> dict:
    """生成-评分-重生成循环（Python while 版）。

    返回 {case: 用例dict, score: judge结果dict, attempts: 实际尝试次数}。
    即使最终未达阈值，也返回最后一版，交给上层决定是否保存。
    """
    llm_gen = get_generation_llm()
    # 首次生成
    resp = llm_gen.invoke(
        [
            SystemMessage(_INIT_PROMPT.format(case_type=case_type)),
            HumanMessage(f"功能点：{point}"),
        ]
    )
    try:
        case = _extract_json(
            resp.content if isinstance(resp.content, str) else str(resp.content)
        )
    except Exception:
        return {
            "case": {"title": "生成失败", "raw": str(resp.content)[:500]},
            "score": {"total": 0, "dims": {}, "comment": "首次生成解析失败", "passed": False},
            "attempts": 1,
        }

    score = judge_case(point, case_type, case)
    attempt = 1
    while not score["passed"] and attempt <= max_retry:
        resp = llm_gen.invoke(
            [
                SystemMessage("你会根据评审反馈重写一条更高质量的测试用例。"),
                HumanMessage(
                    _REGEN_PROMPT.format(
                        point=point,
                        case_type=case_type,
                        last_case=json.dumps(case, ensure_ascii=False, indent=2),
                        feedback=score["comment"],
                    )
                ),
            ]
        )
        try:
            new_case = _extract_json(
                resp.content if isinstance(resp.content, str) else str(resp.content)
            )
        except Exception:
            break  # 解析失败保留上一版
        case = new_case
        score = judge_case(point, case_type, case)
        attempt += 1
    return {"case": case, "score": score, "attempts": attempt}
