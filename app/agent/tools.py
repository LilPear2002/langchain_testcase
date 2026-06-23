"""测试用例生成 Agent 使用的 5 个工具。

使用工厂函数 make_tools(project_id) 把 project_id 绑定进闭包，
避免 LLM 在调用工具时还需要自己传 project_id。
"""
import json
from typing import Callable

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from app.core.llm import get_judge_llm
from app.db import models
from app.db.session import SessionLocal
from app.judge.judge_graph import generate_with_judge_graph
from app.services.vector_store import similarity_search

VALID_CASE_TYPES = {"functional", "boundary", "exception", "negative"}

_POINT_PROMPT = (
    "你是资深测试分析师，请把下面的需求文本拆解成若干可测功能点。"
    "只输出 JSON，结构如下：\n"
    '{"points": [{"name": "功能点名", "desc": "一句话描述"}]}\n'
    "数量建议 3~8 个，粒度适中，避免重复。"
)


def make_tools(project_id: int) -> list[Callable]:

    @tool
    def retrieve_requirement(query: str) -> str:
        """在本项目的需求文档库中检索与 query 最相关的片段，返回拼接后的文本。
        query: 检索关键词或需求主题。"""
        docs = similarity_search(query, project_id=project_id, k=4)
        if not docs:
            return "（未检索到相关需求资料）"
        parts = []
        for i, d in enumerate(docs, 1):
            src = d.metadata.get("doc_name") or "未知"
            parts.append(f"[片段{i} | 来源:{src}]\n{d.page_content}")
        return "\n\n".join(parts)

    @tool
    def analyze_requirement_points(requirement_text: str) -> str:
        """把一段需求文本拆解为若干可测功能点，返回 JSON 字符串。
        requirement_text: 原始需求描述文本。"""
        llm = get_judge_llm()
        resp = llm.invoke(
            [SystemMessage(_POINT_PROMPT), HumanMessage(requirement_text)]
        )
        return resp.content if isinstance(resp.content, str) else str(resp.content)

    @tool
    def generate_test_case(point: str, case_type: str) -> str:
        """针对一个具体功能点生成一条指定类型的测试用例。内部会自动评分，
        若不合格会带反馈重生成（最多 2 次）。返回 JSON，字段包含
        title/preconditions/steps/expected/priority/case_type/score/_judge_comment/_attempts。
        point: 功能点名称或描述。
        case_type: 用例类型，必须是 functional/boundary/exception/negative 之一。"""
        ct = case_type.lower().strip()
        if ct not in VALID_CASE_TYPES:
            return json.dumps(
                {"error": f"case_type 必须是 {sorted(VALID_CASE_TYPES)} 之一"},
                ensure_ascii=False,
            )
        result = generate_with_judge_graph(point, ct)
        case = dict(result["case"])
        case["case_type"] = ct
        case["score"] = result["score"]["total"]
        case["_judge_dims"] = result["score"]["dims"]
        case["_judge_comment"] = result["score"]["comment"]
        case["_judge_passed"] = result["score"]["passed"]
        case["_attempts"] = result["attempts"]
        return json.dumps(case, ensure_ascii=False)

    @tool
    def save_test_cases(cases_json: str) -> str:
        """把一组用例批量保存到数据库。cases_json 必须是 JSON 数组字符串，
        每个对象形如 {title, preconditions, steps, expected, case_type, priority, source?, score?}。
        若对象中带有 generate_test_case 返回的 score 字段，会一并写入。
        返回保存条数、id 列表以及平均分。"""
        try:
            data = json.loads(cases_json)
            if isinstance(data, dict):
                data = data.get("cases") or data.get("items") or [data]
            if not isinstance(data, list):
                return "error: cases_json 必须是数组"
        except json.JSONDecodeError as e:
            return f"error: JSON 解析失败 {e}"

        ids: list[int] = []
        scores: list[int] = []
        with SessionLocal() as db:
            for c in data:
                raw_score = c.get("score")
                score_val = int(raw_score) if isinstance(raw_score, (int, float)) else None
                obj = models.TestCase(
                    project_id=project_id,
                    title=str(c.get("title", "")).strip()[:300] or "未命名用例",
                    preconditions=c.get("preconditions"),
                    steps=c.get("steps") or [],
                    expected=c.get("expected"),
                    case_type=str(c.get("case_type", "functional")).lower(),
                    priority=str(c.get("priority", "P2")).upper(),
                    source=c.get("source"),
                    score=score_val,
                )
                db.add(obj)
                db.flush()
                ids.append(obj.id)
                if score_val is not None:
                    scores.append(score_val)
            db.commit()
        avg = round(sum(scores) / len(scores), 2) if scores else None
        return f"已保存 {len(ids)} 条用例，ids={ids}，平均分={avg}"

    @tool
    def list_existing_cases() -> str:
        """列出本项目已保存的测试用例（仅返回 id 和 title），用于避免重复生成。"""
        with SessionLocal() as db:
            rows = (
                db.query(models.TestCase.id, models.TestCase.title)
                .filter(models.TestCase.project_id == project_id)
                .order_by(models.TestCase.id.desc())
                .limit(50)
                .all()
            )
        if not rows:
            return "（当前项目暂无已保存用例）"
        return json.dumps(
            [{"id": r[0], "title": r[1]} for r in rows], ensure_ascii=False
        )

    return [
        retrieve_requirement,
        analyze_requirement_points,
        generate_test_case,
        save_test_cases,
        list_existing_cases,
    ]
