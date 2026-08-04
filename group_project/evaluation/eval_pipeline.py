"""
RAG Evaluation Pipeline.

Sử dụng DeepEval / RAGAS / TruLens để đánh giá chất lượng RAG pipeline.
Chọn 1 framework và implement đầy đủ.

Yêu cầu:
    1. Load golden_dataset.json (≥15 Q&A pairs)
    2. Chạy RAG pipeline trên từng question
    3. Evaluate với 4 metrics: faithfulness, relevance, context_recall, context_precision
    4. So sánh A/B ít nhất 2 configs
    5. Export results ra results.md

Lưu ý rate limit nếu dùng model OpenRouter ":free": RAGAS/DeepEval gọi LLM RẤT NHIỀU LẦN
(không phải 1 lần/câu hỏi mà nhiều lần/metric/câu hỏi). Model free của OpenRouter giới hạn
50 request/ngày CHO CẢ TÀI KHOẢN (không phải theo model hay theo API key — đổi model free
khác hay tạo key mới KHÔNG reset quota). Nếu chạy full 15+ câu hỏi mà bị rate limit giữa
chừng, thử giảm xuống subset 5 câu để chạy kịp trong buổi, hoặc nạp $10 credit để mở khóa
1000 request/ngày.
"""

import argparse
import json
import os
import sys
from statistics import mean
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"
REQUIRED_GOLDEN_FIELDS = ("question", "expected_answer", "expected_context")
PROJECT_ROOT = Path(__file__).parents[2]
NINE_ROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "http://localhost:20128/v1")
NINE_ROUTER_CHAT_MODEL = os.getenv("OPENROUTER_MODEL", "9router")
# 9Router must expose an embedding-capable model for RAGAS. Override this name
# in .env if the local gateway routes embeddings to a different provider.
# Use the already-downloaded local BGE-M3 embedding model for RAGAS. 9Router
# can serve chat completions, but may not have credentials for an OpenAI
# embedding provider (which would cause HTTP 400 during evaluation).
LOCAL_EMBEDDING_MODEL = os.getenv("RAGAS_EMBEDDING_MODEL", "BAAI/bge-m3")
METRICS = ("faithfulness", "answer_relevancy", "context_recall", "context_precision")

load_dotenv(PROJECT_ROOT / ".env")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_golden_dataset(require_minimum: bool = False) -> list[dict[str, str]]:
    """Load golden dataset từ JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if not isinstance(dataset, list):
        raise ValueError("golden_dataset.json must contain a JSON array.")

    for index, item in enumerate(dataset, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Golden item {index} must be an object.")
        missing = [field for field in REQUIRED_GOLDEN_FIELDS if not item.get(field)]
        if missing:
            raise ValueError(f"Golden item {index} is missing: {', '.join(missing)}")

    if require_minimum and len(dataset) < 15:
        raise ValueError(
            f"Golden dataset needs at least 15 evidence-backed items; found {len(dataset)}."
        )

    return dataset


def _to_eval_row(question: str, expected_answer: str, response: dict[str, Any]) -> dict[str, Any]:
    """Convert the Task 10 response contract into one RAGAS dataset row."""
    if not isinstance(response, dict) or not response.get("answer"):
        raise ValueError("Pipeline response must be a dict containing a non-empty 'answer'.")

    sources = response.get("sources", [])
    if not isinstance(sources, list):
        raise ValueError("Pipeline response field 'sources' must be a list.")

    contexts = [item.get("content", "") for item in sources if isinstance(item, dict)]
    return {
        "question": question,
        "answer": response["answer"],
        "contexts": [context for context in contexts if context],
        "ground_truth": expected_answer,
        "retrieval_source": response.get("retrieval_source", "unknown"),
    }


def run_task10_with_config(question: str, config: dict[str, Any]) -> dict[str, Any]:
    """A/B adapter from this evaluator to the Task 10 generation contract.

    ``config`` is deliberately limited to Task 10 keyword arguments. The
    default comparison only sends ``use_reranking`` to Task 9 via Task 10.
    """
    from src.task10_generation import generate_with_citation

    return generate_with_citation(question, **config)


def build_9router_ragas_models() -> tuple[Any, Any]:
    """Use 9Router for LLM judging and local BGE-M3 for embeddings."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY in .env for 9Router first.")

    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper
    except ImportError as exc:
        raise RuntimeError(
            "Install ragas, datasets, and langchain-openai before running evaluation."
        ) from exc

    llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=NINE_ROUTER_CHAT_MODEL,
            api_key=api_key,
            base_url=NINE_ROUTER_BASE_URL,
        )
    )
    embeddings = HuggingFaceEmbeddings(
        model_name=LOCAL_EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return llm, embeddings


# =============================================================================
# Option 1: DeepEval
# =============================================================================

def evaluate_with_deepeval(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng DeepEval.

    pip install deepeval
    """
    # TODO: Implement
    #
    # from deepeval import evaluate
    # from deepeval.metrics import (
    #     FaithfulnessMetric,
    #     AnswerRelevancyMetric,
    #     ContextualRecallMetric,
    #     ContextualPrecisionMetric,
    # )
    # from deepeval.test_case import LLMTestCase
    #
    # test_cases = []
    # for item in golden_dataset:
    #     result = rag_pipeline.generate_with_citation(item["question"])
    #     test_case = LLMTestCase(
    #         input=item["question"],
    #         actual_output=result["answer"],
    #         expected_output=item["expected_answer"],
    #         retrieval_context=[c["content"] for c in result["sources"]],
    #     )
    #     test_cases.append(test_case)
    #
    # metrics = [
    #     FaithfulnessMetric(threshold=0.7),
    #     AnswerRelevancyMetric(threshold=0.7),
    #     ContextualRecallMetric(threshold=0.7),
    #     ContextualPrecisionMetric(threshold=0.7),
    # ]
    #
    # results = evaluate(test_cases, metrics)
    # return results
    raise NotImplementedError("Implement evaluate_with_deepeval")


# =============================================================================
# Option 2: RAGAS
# =============================================================================

def evaluate_with_ragas(
    rag_pipeline: Callable[[str], dict[str, Any]],
    golden_dataset: list[dict],
    llm: Any | None = None,
    embeddings: Any | None = None,
) -> dict[str, Any]:
    """
    Evaluate RAG pipeline sử dụng RAGAS.

    pip install ragas
    """
    # ``rag_pipeline`` must be a callable compatible with Task 10:
    # generate_with_citation(question) -> {"answer": str, "sources": list[dict]}.
    #
    # from ragas import evaluate
    # from ragas.metrics import (
    #     faithfulness,
    #     answer_relevancy,
    #     context_recall,
    #     context_precision,
    # )
    # from datasets import Dataset
    #
    # eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    #
    # for item in golden_dataset:
    #     result = rag_pipeline.generate_with_citation(item["question"])
    #     eval_data["question"].append(item["question"])
    #     eval_data["answer"].append(result["answer"])
    #     eval_data["contexts"].append([c["content"] for c in result["sources"]])
    #     eval_data["ground_truth"].append(item["expected_answer"])
    #
    # dataset = Dataset.from_dict(eval_data)
    # result = evaluate(
    #     dataset,
    #     metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    # )
    # return result.to_pandas()
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from ragas.run_config import RunConfig
    except ImportError as exc:
        raise RuntimeError(
            "Install the evaluation dependencies first: "
            "pip install 'ragas==0.1.21' 'datasets>=2.14.0' 'langchain-openai>=0.1.0'"
        ) from exc

    if not callable(rag_pipeline):
        raise TypeError("rag_pipeline must be a callable such as generate_with_citation.")

    rows = [
        _to_eval_row(item["question"], item["expected_answer"], rag_pipeline(item["question"]))
        for item in golden_dataset
    ]
    if llm is None or embeddings is None:
        llm, embeddings = build_9router_ragas_models()

    # Gemini's OpenAI-compatible endpoint accepts a single completion per call.
    # RAGAS AnswerRelevancy defaults to strictness=3, which requests multiple
    # candidates and triggers HTTP 400. One candidate preserves the metric while
    # remaining compatible with Gemini.
    answer_relevancy.strictness = 1

    result = evaluate(
        Dataset.from_list(rows),
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm,
        embeddings=embeddings,
        # Gemini free tier allows 15 generate requests/minute. Sequential jobs
        # plus retries are slower but avoid invalid parallel bursts and preserve
        # a complete, comparable A/B run.
        run_config=RunConfig(max_workers=1, max_retries=10, max_wait=90),
    )
    case_rows = result.to_pandas().to_dict(orient="records")
    return {
        "summary": {name: float(result[name]) for name in METRICS if name in result},
        "cases": case_rows,
    }


# =============================================================================
# Option 3: TruLens
# =============================================================================

def evaluate_with_trulens(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng TruLens.

    pip install trulens
    """
    # TODO: Implement
    #
    # from trulens.apps.custom import TruCustomApp
    # from trulens.core import Feedback
    # from trulens.providers.openai import OpenAI as TruOpenAI
    #
    # provider = TruOpenAI()
    #
    # f_faithfulness = Feedback(provider.groundedness_measure_with_cot_reasons).on_output()
    # f_relevance = Feedback(provider.relevance).on_input_output()
    # f_context_relevance = Feedback(provider.context_relevance).on_input()
    #
    # tru_rag = TruCustomApp(
    #     rag_pipeline,
    #     app_name="UniversityServices_RAG",
    #     feedbacks=[f_faithfulness, f_relevance, f_context_relevance],
    # )
    #
    # with tru_rag as recording:
    #     for item in golden_dataset:
    #         rag_pipeline.generate_with_citation(item["question"])
    #
    # # Dashboard: from trulens.dashboard import run_dashboard; run_dashboard()
    raise NotImplementedError("Implement evaluate_with_trulens")


# =============================================================================
# A/B Comparison
# =============================================================================

def compare_configs(
    configured_pipeline: Callable[[str, dict[str, Any]], dict[str, Any]],
    golden_dataset: list[dict],
    configs: dict[str, dict[str, Any]] | None = None,
    llm: Any | None = None,
    embeddings: Any | None = None,
) -> dict[str, dict]:
    """
    So sánh A/B giữa ít nhất 2 configs.

    Gợi ý configs để so sánh:
    - Config A: hybrid search + reranking
    - Config B: hybrid search without reranking
    - Config C: hybrid search + PageIndex fallback
    """
    # ``configured_pipeline`` is the integration hand-off from the Task 9 owner:
    # configured_pipeline(question, config) -> Task 10 response contract.
    #
    # configs = {
    #     "hybrid_rerank": {"use_reranking": True, "alpha": 0.5},
    #     "dense_only": {"use_reranking": False, "alpha": 1.0},
    # }
    #
    # results = {}
    # for config_name, params in configs.items():
    #     # Run eval with this config
    #     ...
    #     results[config_name] = scores
    #
    # return results
    if not callable(configured_pipeline):
        raise TypeError("configured_pipeline must accept (question, config).")

    configs = configs or {
        "hybrid_rerank": {"use_reranking": True},
        "hybrid_no_rerank": {"use_reranking": False},
    }
    if llm is None or embeddings is None:
        llm, embeddings = build_9router_ragas_models()

    comparison: dict[str, dict] = {}
    for name, config in configs.items():
        response_fn = lambda question, active_config=config: configured_pipeline(
            question, active_config
        )
        comparison[name] = evaluate_with_ragas(
            response_fn, golden_dataset, llm=llm, embeddings=embeddings
        )
    return comparison


# =============================================================================
# Export Results
# =============================================================================

def _score(row: dict[str, Any]) -> float:
    values = [float(row[name]) for name in METRICS if isinstance(row.get(name), (int, float))]
    return mean(values) if values else 0.0


def _root_cause(row: dict[str, Any]) -> str:
    scores = {name: float(row[name]) for name in METRICS if isinstance(row.get(name), (int, float))}
    if not scores:
        return "Metric row missing or evaluation failed"
    weakest = min(scores, key=scores.get)
    return {
        "context_recall": "Retriever missed expected evidence",
        "context_precision": "Retrieved context contains too much noise",
        "faithfulness": "Generation is insufficiently grounded in context",
        "answer_relevancy": "Answer is not sufficiently focused on the question",
    }[weakest]


def export_results(comparison: dict[str, dict[str, Any]]) -> None:
    """Export evaluation results to results.md"""
    # TODO: Format and write results
    #
    # content = "# RAG Evaluation Results\n\n"
    # content += "## Overall Scores\n\n"
    # content += "| Metric | Score |\n|--------|-------|\n"
    # ...
    # content += "\n## A/B Comparison\n\n"
    # ...
    # content += "\n## Worst Performers\n\n"
    # ...
    # content += "\n## Recommendations\n\n"
    # ...
    #
    # RESULTS_PATH.write_text(content, encoding="utf-8")
    if not comparison:
        raise ValueError("No A/B results available to export.")

    config_scores = {
        name: mean(list(payload["summary"].values())) if payload["summary"] else 0.0
        for name, payload in comparison.items()
    }
    winner = max(config_scores, key=config_scores.get)
    names = list(comparison)
    left, right = names[0], names[1] if len(names) > 1 else names[0]

    content = "# RAG Evaluation Results\n\n"
    content += "## Framework and setup\n\n"
    content += "- Framework: RAGAS\n"
    content += f"- Dataset: {len(load_golden_dataset(require_minimum=True))} evidence-backed questions\n"
    content += f"- Config A: `{left}`\n- Config B: `{right}`\n"
    content += (
        f"- Evaluator: 9Router model `{NINE_ROUTER_CHAT_MODEL}`; embeddings: "
        f"local `{LOCAL_EMBEDDING_MODEL}`\n\n"
    )
    content += "## Overall scores\n\n"
    content += f"| Metric | {left} | {right} | Delta (A-B) |\n"
    content += "|---|---:|---:|---:|\n"
    for metric in METRICS:
        a = comparison[left]["summary"].get(metric, 0.0)
        b = comparison[right]["summary"].get(metric, 0.0)
        content += f"| {metric} | {a:.3f} | {b:.3f} | {a - b:+.3f} |\n"
    content += (
        f"| **Average** | **{config_scores[left]:.3f}** | "
        f"**{config_scores[right]:.3f}** | **{config_scores[left] - config_scores[right]:+.3f}** |\n\n"
    )
    content += "## A/B conclusion\n\n"
    content += (
        f"`{winner}` is the winner by mean of the four RAGAS metrics "
        f"({config_scores[winner]:.3f}). This conclusion is based on this fixed 15-question dataset; "
        "rerun it after changing corpus, chunking, retrieval, or model settings.\n\n"
    )

    all_cases = []
    for config, payload in comparison.items():
        for row in payload["cases"]:
            all_cases.append((config, row, _score(row)))
    worst_cases = sorted(all_cases, key=lambda item: item[2])[:3]
    content += "## Worst performers (bottom 3)\n\n"
    content += "| Config | Question | Average metric | Root cause |\n|---|---|---:|---|\n"
    for config, row, score in worst_cases:
        question = str(row.get("question", "")).replace("|", "\\|")
        content += f"| {config} | {question} | {score:.3f} | {_root_cause(row)} |\n"

    content += "\n## Recommendations\n\n"
    content += "1. Inspect the bottom-three queries and add or correct source chunks before changing prompts.\n"
    content += "2. If context precision is weak, reduce candidate noise with tighter chunking or stronger reranking.\n"
    content += "3. If faithfulness is weak, require citation for every factual claim and refuse answers with insufficient evidence.\n"
    RESULTS_PATH.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the 15-question RAGAS A/B evaluation.")
    parser.add_argument("--run", action="store_true", help="Call Gemini and write results.md")
    args = parser.parse_args()
    golden_dataset = load_golden_dataset(require_minimum=True)
    print(f"Loaded {len(golden_dataset)} test cases")

    # TODO: Import your RAG pipeline
    # from src.task10_generation import generate_with_citation
    #
    # Chọn 1 framework:
    # results = evaluate_with_deepeval(pipeline, golden_dataset)
    # results = evaluate_with_ragas(pipeline, golden_dataset)
    # results = evaluate_with_trulens(pipeline, golden_dataset)
    #
    # comparison = compare_configs(run_task10_with_config, golden_dataset)
    # export_results(comparison)
    if args.run:
        comparison = compare_configs(run_task10_with_config, golden_dataset)
        export_results(comparison)
        print(f"Evaluation complete. Report written to {RESULTS_PATH}")
    else:
        print("Ready. Run with --run to spend API quota and generate results.md.")
