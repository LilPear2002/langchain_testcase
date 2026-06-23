"""从某项目已索引文档自动生成 RAG 评测集（question + reference）。

用一次受控的结构化 prompt 让 qwen 基于文档内容生成测试集——比 RAGAS
TestsetGenerator 的知识图谱流水线在小型中文文档上快得多、稳得多。
生成的 reference 完全取自文档原文，可支撑后续 RAGAS 的召回/精确率评测。

运行（在 backend 目录下）：
    conda run -n fastapi python eval/gen_testset.py --project-id 2 --size 6

产出 eval/dataset.json：[{"question": ..., "reference": ...}, ...]
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.llm import get_judge_llm  # noqa: E402
from app.services.vector_store import get_vector_store  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
DATASET_PATH = EVAL_DIR / "dataset.json"

PROMPT = """你是 RAG 系统的测评数据构造专家。下面是某需求文档的全部内容，请基于\
它生成 {n} 条高质量的「问题-标准答案」对，用于评测检索增强问答系统。

要求：
1. 问题要具体、可由文档明确回答，覆盖文档的不同知识点，避免雷同。
2. 标准答案（reference）必须完全依据文档原文，不得编造或添加文档外信息。
3. 答案简洁准确，直接陈述事实。
4. 严格输出 JSON 数组，每个元素形如 {{"question": "...", "reference": "..."}}，\
不要输出任何额外说明或 markdown 代码块标记。

文档内容：
\"\"\"
{content}
\"\"\"
"""


def load_project_content(project_id: int) -> str:
    """从 Chroma 取出该项目所有 chunk，拼成完整文档内容。"""
    vs = get_vector_store()
    r = vs._collection.get(where={"project_id": project_id})
    contents = r.get("documents") or []
    return "\n\n".join(contents)


def _extract_json(text: str) -> list[dict]:
    """从模型输出里抽取 JSON 数组，容忍 ```json 包裹等情况。"""
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1:
        text = text[start : end + 1]
    return json.loads(text)


def main(project_id: int, size: int) -> None:
    content = load_project_content(project_id)
    print(f"[load] project={project_id} chars={len(content)}")
    if not content.strip():
        raise SystemExit("该项目没有已索引文档，先在前端上传并向量化。")

    llm = get_judge_llm()
    msg = PROMPT.format(n=size, content=content)
    print("[gen] 调用 qwen 生成测试集 ...")
    resp = llm.invoke(msg).content
    if isinstance(resp, list):
        resp = "".join(str(p) for p in resp)

    try:
        items = _extract_json(resp)
    except json.JSONDecodeError as e:
        print("[error] 解析模型输出失败，原始输出如下：")
        print(resp)
        raise SystemExit(str(e))

    rows: list[dict] = []
    for it in items:
        q = (it.get("question") or "").strip()
        ref = (it.get("reference") or "").strip()
        if q and ref:
            rows.append({"question": q, "reference": ref})

    DATASET_PATH.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[save] {len(rows)} 条 -> {DATASET_PATH}")
    for i, r in enumerate(rows, 1):
        print(f"  Q{i}: {r['question']}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--project-id", type=int, default=2)
    p.add_argument("--size", type=int, default=6)
    args = p.parse_args()
    main(args.project_id, args.size)
