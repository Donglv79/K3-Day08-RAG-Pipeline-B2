"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

# Cache để không tạo lại BM25 ở mỗi lần query. Corpus dùng chunks giống Task 4,
# nhờ vậy dense và sparse trả về cùng mức chi tiết cho Task 9 fusion.
CORPUS: list[dict] = []
_bm25_index: BM25Okapi | None = None


def _tokenize(text: str) -> list[str]:
    """Tokenize nhẹ, hỗ trợ chữ Unicode (Việt/Anh) và bỏ dấu câu."""
    return re.findall(r"[^\W_]+", text.lower(), flags=re.UNICODE)


def _load_corpus() -> list[dict]:
    """Nạp Markdown, ưu tiên chunks Task 4; fallback về full document nếu cần."""
    global CORPUS
    if CORPUS:
        return CORPUS

    try:
        from .task4_chunking_indexing import chunk_documents, load_documents

        documents = load_documents()
        CORPUS = chunk_documents(documents) if documents else []
    except Exception:
        # BM25 vẫn hoạt động khi Task 4/chunking dependency chưa được cài.
        for md_file in STANDARDIZED_DIR.rglob("*.md"):
            content = md_file.read_text(encoding="utf-8").strip()
            if content:
                doc_type = "legal" if md_file.parent.name == "legal" else "news"
                CORPUS.append({
                    "content": content,
                    "metadata": {"source": md_file.name, "type": doc_type},
                })
    return CORPUS


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    tokenized_corpus = [_tokenize(doc.get("content", "")) for doc in corpus]
    # BM25Okapi không nhận corpus rỗng; caller xử lý trường hợp không có data.
    return BM25Okapi(tokenized_corpus) if tokenized_corpus else None


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    global _bm25_index
    if top_k <= 0 or not query.strip():
        return []

    corpus = _load_corpus()
    if not corpus:
        return []
    if _bm25_index is None:
        _bm25_index = build_bm25_index(corpus)
    if _bm25_index is None:
        return []

    scores = _bm25_index.get_scores(_tokenize(query))
    # Stable sort: điểm bằng nhau giữ thứ tự corpus để kết quả tái lập được.
    top_indices = np.argsort(-scores, kind="stable")[:min(top_k, len(corpus))]
    results = []
    for index in top_indices:
        score = float(scores[index])
        if score <= 0:
            continue
        doc = corpus[int(index)]
        results.append({
            "content": doc["content"],
            "score": round(score, 4),
            "metadata": dict(doc.get("metadata", {})),
        })
    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("tuition fee payment methods", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
