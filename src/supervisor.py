"""
Advanced pattern: Supervisor + parallel workers.

This module is intentionally separate from Task 9 so the graded retrieval
pipeline stays stable. It adds orchestration, parallel dense/sparse retrieval,
trace metadata, and graceful worker-level error handling for demos.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from .task5_semantic_search import hyde_search, semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search
from .task9_retrieval_pipeline import DEFAULT_TOP_K, RERANK_METHOD, SCORE_THRESHOLD


@dataclass
class WorkerResult:
    name: str
    results: list[dict]
    elapsed_ms: float
    error: str | None = None


class DenseWorker:
    name = "dense"

    def __init__(self, use_hyde: bool = False):
        self.use_hyde = use_hyde

    def run(self, query: str, top_k: int) -> list[dict]:
        search_fn = hyde_search if self.use_hyde else semantic_search
        return search_fn(query, top_k=top_k)


class LexicalWorker:
    name = "lexical"

    def run(self, query: str, top_k: int) -> list[dict]:
        return lexical_search(query, top_k=top_k)


class FusionWorker:
    name = "fusion"

    def run(self, ranked_lists: list[list[dict]], top_k: int) -> list[dict]:
        merged = rerank_rrf(ranked_lists, top_k=top_k) if ranked_lists else []
        return [
            {
                **item,
                "metadata": dict(item.get("metadata", {})),
                "source": "hybrid",
            }
            for item in merged
        ]


class RerankWorker:
    name = "rerank"

    def __init__(self, method: str = RERANK_METHOD):
        self.method = method

    def run(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        return rerank(query, candidates, top_k=top_k, method=self.method)


class FallbackWorker:
    name = "pageindex"

    def run(self, query: str, top_k: int) -> list[dict]:
        return pageindex_search(query, top_k=top_k)


def _run_timed(name: str, fn, *args: Any, **kwargs: Any) -> WorkerResult:
    start = perf_counter()
    try:
        results = fn(*args, **kwargs)
        error = None
    except Exception as exc:
        results = []
        error = f"{exc.__class__.__name__}: {exc}"
    elapsed_ms = (perf_counter() - start) * 1000
    return WorkerResult(name=name, results=results, elapsed_ms=elapsed_ms, error=error)


class RetrievalSupervisor:
    """
    Coordinates retrieval workers and exposes trace data for debugging/demo.

    Dense and lexical search run concurrently because they are independent.
    Fusion, reranking, and fallback are sequential because each depends on the
    previous stage's output.
    """

    def __init__(
        self,
        score_threshold: float = SCORE_THRESHOLD,
        rerank_method: str = RERANK_METHOD,
        max_workers: int = 2,
    ):
        self.score_threshold = score_threshold
        self.rerank_method = rerank_method
        self.max_workers = max_workers

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        score_threshold: float | None = None,
        use_hyde: bool = False,
        use_reranking: bool = True,
    ) -> dict:
        if top_k <= 0 or not query.strip():
            return {
                "results": [],
                "trace": {"reason": "empty_query_or_top_k"},
                "fallback_used": False,
                "errors": {},
            }

        threshold = self.score_threshold if score_threshold is None else score_threshold
        candidate_k = max(top_k * 2, top_k)
        errors: dict[str, str] = {}
        timings_ms: dict[str, float] = {}

        dense_worker = DenseWorker(use_hyde=use_hyde)
        lexical_worker = LexicalWorker()
        parallel_results: dict[str, list[dict]] = {"dense": [], "lexical": []}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    _run_timed, dense_worker.name, dense_worker.run, query, candidate_k
                ): dense_worker.name,
                executor.submit(
                    _run_timed, lexical_worker.name, lexical_worker.run, query, candidate_k
                ): lexical_worker.name,
            }
            for future in as_completed(futures):
                worker_result = future.result()
                parallel_results[worker_result.name] = worker_result.results
                timings_ms[worker_result.name] = round(worker_result.elapsed_ms, 2)
                if worker_result.error:
                    errors[worker_result.name] = worker_result.error

        dense_results = parallel_results["dense"]
        lexical_results = parallel_results["lexical"]
        best_semantic_score = (
            float(dense_results[0].get("score", 0.0)) if dense_results else 0.0
        )

        fusion_result = _run_timed(
            FusionWorker.name,
            FusionWorker().run,
            [items for items in (dense_results, lexical_results) if items],
            candidate_k,
        )
        timings_ms[fusion_result.name] = round(fusion_result.elapsed_ms, 2)
        if fusion_result.error:
            errors[fusion_result.name] = fusion_result.error
        merged = fusion_result.results

        if use_reranking and merged:
            rerank_worker = RerankWorker(method=self.rerank_method)
            rerank_result = _run_timed(
                rerank_worker.name, rerank_worker.run, query, merged, top_k
            )
            timings_ms[rerank_result.name] = round(rerank_result.elapsed_ms, 2)
            if rerank_result.error:
                errors[rerank_result.name] = rerank_result.error
                final_results = merged[:top_k]
            else:
                final_results = rerank_result.results
        else:
            final_results = merged[:top_k]

        fallback_attempted = best_semantic_score < threshold
        fallback_used = False
        if fallback_attempted:
            fallback_worker = FallbackWorker()
            fallback_result = _run_timed(
                fallback_worker.name, fallback_worker.run, query, top_k
            )
            timings_ms[fallback_result.name] = round(fallback_result.elapsed_ms, 2)
            if fallback_result.error:
                errors[fallback_result.name] = fallback_result.error
            if fallback_result.results:
                fallback_used = True
                final_results = [
                    {
                        **item,
                        "metadata": dict(item.get("metadata", {})),
                        "source": "pageindex",
                    }
                    for item in fallback_result.results[:top_k]
                ]

        final_results = [
            {
                **item,
                "metadata": dict(item.get("metadata", {})),
                "source": item.get("source", "hybrid"),
            }
            for item in final_results[:top_k]
        ]

        trace = {
            "query": query,
            "top_k": top_k,
            "candidate_k": candidate_k,
            "use_hyde": use_hyde,
            "use_reranking": use_reranking,
            "rerank_method": self.rerank_method,
            "score_threshold": threshold,
            "best_semantic_score": round(best_semantic_score, 4),
            "dense_count": len(dense_results),
            "lexical_count": len(lexical_results),
            "merged_count": len(merged),
            "final_count": len(final_results),
            "fallback_attempted": fallback_attempted,
            "fallback_used": fallback_used,
            "timings_ms": timings_ms,
        }

        return {
            "results": final_results,
            "trace": trace,
            "fallback_used": fallback_used,
            "errors": errors,
        }


def retrieve_supervised(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_hyde: bool = False,
    use_reranking: bool = True,
) -> dict:
    """Convenience API for the advanced Supervisor + Workers pattern."""
    supervisor = RetrievalSupervisor(score_threshold=score_threshold)
    return supervisor.retrieve(
        query=query,
        top_k=top_k,
        score_threshold=score_threshold,
        use_hyde=use_hyde,
        use_reranking=use_reranking,
    )


def generate_supervised(query: str, top_k: int = DEFAULT_TOP_K) -> dict:
    """
    Generation wrapper that reuses supervised retrieval and Task 10 formatting.

    This avoids a second retrieval call while keeping Task 10's prompt and LLM
    provider chain as the single source of truth.
    """
    from .task10_generation import SYSTEM_PROMPT, _call_llm, format_context, reorder_for_llm

    retrieval = retrieve_supervised(query=query, top_k=top_k)
    chunks = retrieval["results"]
    if not chunks:
        return {
            "answer": "I cannot verify this information",
            "sources": [],
            "retrieval_source": "none",
            "trace": retrieval["trace"],
            "errors": retrieval["errors"],
        }

    context = format_context(reorder_for_llm(chunks))
    answer = _call_llm([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\n---\n\nQuestion: {query}"},
    ])
    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid"),
        "trace": retrieval["trace"],
        "errors": retrieval["errors"],
    }


if __name__ == "__main__":
    demo = retrieve_supervised("tuition fee payment", top_k=3)
    print("Results:", len(demo["results"]))
    print("Trace:", demo["trace"])
    print("Errors:", demo["errors"])
