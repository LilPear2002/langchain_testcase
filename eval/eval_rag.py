"""用 RAGAS 评测某项目 RAG 链路的四个核心指标。

流程：读 eval/dataset.json → 对每个问题复用项目的 similarity_search 取检索片段、
跑一次精简 RAG（无历史）拿回答 → 组成 EvaluationDataset → RAGAS evaluate。

裁判 LLM 复用 qwen（judge_llm），embedding 复用 text-embedding-v4。

运行（在 backend 目录下）：
    conda run -n fastapi python eval/eval_rag.py --project-id 2 --k 4
"""
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402

from app.core.llm import get_embeddings, get_generation_llm, get_judge_llm  # noqa: E402
from app.rag.rag_chain import SYSTEM_PROMPT, _format_docs  # noqa: E402
from app.services.vector_store import similarity_search  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
DATASET_PATH = EVAL_DIR / "dataset.json"


def build_answer_chain():
    """精简版 RAG 链（无历史），与生产 prompt 一致。"""
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", "{input}")]
    )
    return prompt | get_generation_llm() | StrOutputParser()


def main(project_id: int, k: int) -> None:
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    print(f"[load] {len(data)} 条样本，project={project_id}, k={k}")

    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )
    from ragas.run_config import RunConfig

    chain = build_answer_chain()
    samples: list[SingleTurnSample] = []
    for i, item in enumerate(data, 1):
        q = item["question"]
        docs = similarity_search(q, project_id=project_id, k=k)
        contexts = [d.page_content for d in docs]
        answer = chain.invoke({"input": q, "context": _format_docs(docs)})
        print(f"  [{i}/{len(data)}] 已生成回答（检索 {len(contexts)} 片段）")
        samples.append(
            SingleTurnSample(
                user_input=q,
                response=answer,
                retrieved_contexts=contexts,
                reference=item["reference"],
            )
        )

    dataset = EvaluationDataset(samples=samples)
    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
    ]

    print("[eval] 调用 RAGAS 评测中（裁判=qwen，低并发以避免风控）...")
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=get_judge_llm(),
        embeddings=get_embeddings(),
        run_config=RunConfig(max_workers=2, timeout=180),
        raise_exceptions=False,
        show_progress=True,
    )

    df = result.to_pandas()
    metric_names = [m.name for m in metrics]
    print("\n=== RAGAS 评测结果（均值）===")
    for name in metric_names:
        if name in df.columns:
            print(f"  {name:38s}: {df[name].mean():.4f}")

    out = EVAL_DIR / f"results_p{project_id}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n[save] 明细 -> {out}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--project-id", type=int, default=2)
    p.add_argument("--k", type=int, default=4)
    args = p.parse_args()
    main(args.project_id, args.k)
