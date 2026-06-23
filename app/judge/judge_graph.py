"""LangGraph 版本的『生成-评分-重生成』状态机。

把 judge.py 里的 Python while 循环升级为 StateGraph：

    START ──► generate ──► judge ──┐
                               ▲    │
                               │    ├─ passed or attempt>=max ─► END
                               │    │
                           regenerate ◄──┘

节点之间用 Command(goto=..., update=...) 同时完成路由和状态更新。
"""
from __future__ import annotations

import json
import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from app.core.llm import get_generation_llm
from app.judge.judge import (
    MAX_RETRY,
    PASS_THRESHOLD,
    _INIT_PROMPT,
    _REGEN_PROMPT,
    _extract_json,
    judge_case,
)


class JudgeState(TypedDict, total=False):
    # 输入
    point: str
    case_type: str
    threshold: int
    max_retry: int
    # 过程
    case: dict[str, Any] | None
    score: dict[str, Any] | None
    attempt: int
    # Debug 轨迹（reducer 为 list.add，每个节点追加一条）
    trace: Annotated[list[dict[str, Any]], operator.add]


def _gen_initial_case(point: str, case_type: str) -> dict[str, Any]:
    llm = get_generation_llm()
    resp = llm.invoke(
        [
            SystemMessage(_INIT_PROMPT.format(case_type=case_type)),
            HumanMessage(f"功能点：{point}"),
        ]
    )
    raw = resp.content if isinstance(resp.content, str) else str(resp.content)
    try:
        return _extract_json(raw)
    except Exception:
        return {"title": "生成失败", "raw": raw[:500], "_parse_error": True}


def _regen_with_feedback(
    point: str, case_type: str, last_case: dict, feedback: str
) -> dict[str, Any] | None:
    llm = get_generation_llm()
    resp = llm.invoke(
        [
            SystemMessage("你会根据评审反馈重写一条更高质量的测试用例。"),
            HumanMessage(
                _REGEN_PROMPT.format(
                    point=point,
                    case_type=case_type,
                    last_case=json.dumps(last_case, ensure_ascii=False, indent=2),
                    feedback=feedback,
                )
            ),
        ]
    )
    raw = resp.content if isinstance(resp.content, str) else str(resp.content)
    try:
        return _extract_json(raw)
    except Exception:
        return None  # 解析失败则保留上一版


# ---------- 节点 ----------

def generate_node(state: JudgeState) -> Command[Literal["judge"]]:
    case = _gen_initial_case(state["point"], state["case_type"])
    step = {
        "node": "generate",
        "attempt": 1,
        "case_title": case.get("title"),
    }
    return Command(goto="judge", update={"case": case, "attempt": 1, "trace": [step]})


def judge_node(state: JudgeState) -> Command[Literal["regenerate", "__end__"]]:
    case = state["case"] or {}
    score = judge_case(state["point"], state["case_type"], case)
    threshold = state.get("threshold", PASS_THRESHOLD)
    max_retry = state.get("max_retry", MAX_RETRY)
    attempt = state.get("attempt", 1)
    step = {
        "node": "judge",
        "attempt": attempt,
        "total": score["total"],
        "passed": score["total"] >= threshold,
        "comment": score["comment"][:120],
    }
    # passed 字段直接用阈值重算，避免 judge_case 内默认阈值与调用方不一致
    score["passed"] = score["total"] >= threshold
    if score["passed"] or attempt >= max_retry + 1:
        return Command(goto=END, update={"score": score, "trace": [step]})
    return Command(goto="regenerate", update={"score": score, "trace": [step]})


def regenerate_node(state: JudgeState) -> Command[Literal["judge"]]:
    attempt = state.get("attempt", 1) + 1
    new_case = _regen_with_feedback(
        state["point"],
        state["case_type"],
        state["case"] or {},
        (state.get("score") or {}).get("comment", ""),
    )
    # 解析失败：保留上一版 case，直接跳到 END 避免死循环
    if new_case is None:
        step = {"node": "regenerate", "attempt": attempt, "parse_error": True}
        return Command(goto=END, update={"trace": [step]})
    step = {
        "node": "regenerate",
        "attempt": attempt,
        "case_title": new_case.get("title"),
    }
    return Command(
        goto="judge",
        update={"case": new_case, "attempt": attempt, "trace": [step]},
    )


# ---------- 编译 ----------

def _build_graph():
    g = StateGraph(JudgeState)
    g.add_node("generate", generate_node)
    g.add_node("judge", judge_node)
    g.add_node("regenerate", regenerate_node)
    g.add_edge(START, "generate")
    return g.compile()


_graph = None


def get_judge_graph():
    """惰性编译并缓存图实例。"""
    global _graph
    if _graph is None:
        _graph = _build_graph()
    return _graph


# ---------- 对外接口 ----------

def generate_with_judge_graph(
    point: str,
    case_type: str,
    max_retry: int = MAX_RETRY,
    threshold: int = PASS_THRESHOLD,
    return_trace: bool = False,
) -> dict[str, Any]:
    """与 judge.generate_with_judge 等价的入口，但底层用 StateGraph 驱动。

    return_trace=True 时返回值多一个 ``trace`` 字段，列出每个节点的状态快照，
    供 debug 接口使用。
    """
    graph = get_judge_graph()
    final: JudgeState = graph.invoke(
        {
            "point": point,
            "case_type": case_type,
            "threshold": threshold,
            "max_retry": max_retry,
            "attempt": 0,
            "trace": [],
        }
    )
    out: dict[str, Any] = {
        "case": final.get("case") or {},
        "score": final.get("score")
        or {"total": 0, "dims": {}, "comment": "未产生评分", "passed": False},
        "attempts": final.get("attempt", 1),
    }
    if return_trace:
        out["trace"] = final.get("trace", [])
    return out


def render_mermaid() -> str:
    """返回编译后图的 Mermaid 源码。"""
    return get_judge_graph().get_graph().draw_mermaid()
