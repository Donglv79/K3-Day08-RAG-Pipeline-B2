"""
Task 5 — Semantic Search Module (Dense Retrieval).

    semantic_search(query, top_k): embed query bằng cùng model ở Task 4
    (BAAI/bge-m3), query ChromaDB theo cosine similarity, trả về chunks kèm
    cosine score giảm dần theo schema chung {'content','score','metadata'}.

Bonus — HyDE (Hypothetical Document Embeddings):
    hyde_search(query, top_k): dùng LLM (OpenRouter) sinh một "câu trả lời giả
    định" cho query, sau đó embed đoạn đó để truy vấn — tăng recall với câu hỏi
    dùng từ ngữ khác tài liệu. Thiếu OPENROUTER_API_KEY thì tự fallback về
    semantic_search (không crash).
"""

import os
from pathlib import Path

from .task4_chunking_indexing import get_collection, get_embedding_model

PROJECT_DIR = Path(__file__).parent.parent
HYDE_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
HYDE_SYSTEM_PROMPT = (
    "You are a retrieval assistant. Write a short, factual passage (max 150 words) "
    "that would appear in an official university document answering the user's "
    "question. Use only general, verifiable-sounding information; do not invent "
    "specific numbers. If you cannot answer, write 'I do not have information.'"
)


def _load_env_key(key: str) -> str:
    """Đọc API key từ biến môi trường hoặc file .env (không cần python-dotenv)."""
    value = os.environ.get(key, "")
    if value:
        return value
    env_path = PROJECT_DIR / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                if k.strip() == key:
                    return v.strip().strip('"').strip("'")
    return ""


def hypothetical_document(query: str) -> str:
    """
    Dùng LLM (OpenRouter) sinh hypothetical document cho query (HyDE).

    Returns:
        Chuỗi văn bản giả định, hoặc "" nếu thiếu API key / gọi LLM lỗi.
    """
    api_key = _load_env_key("OPENROUTER_API_KEY")
    if not api_key:
        print("  ⚠ Không có OPENROUTER_API_KEY — bỏ qua HyDE")
        return ""

    try:
        import requests

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": HYDE_MODEL,
                "messages": [
                    {"role": "system", "content": HYDE_SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                "temperature": 0.3,
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"]["content"].strip()
        return text
    except Exception as e:
        print(f"  ⚠ HyDE LLM thất bại ({e.__class__.__name__}: {e}) — dùng query gốc")
        return ""


def _search_with_embedding(query_embedding: list[float], top_k: int) -> list[dict]:
    """Query ChromaDB bằng vector có sẵn, trả về schema chuẩn sorted desc."""
    collection = get_collection()
    count = collection.count()
    if count == 0:
        print("  ⚠ Collection rỗng — chạy Task 4 trước đã")
        return []

    n_results = min(top_k, count)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        # Chroma dùng space cosine → distance = 1 - cosine_similarity
        score = max(0.0, 1.0 - dist)
        output.append({
            "content": doc,
            "score": round(score, 4),
            "metadata": meta,
        })

    output.sort(key=lambda x: x["score"], reverse=True)
    return output[:top_k]


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
        Sorted by score (cosine) descending.
    """
    model = get_embedding_model()
    query_vector = model.encode(query).tolist()
    return _search_with_embedding(query_vector, top_k)


def hyde_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Bonus HyDE: tạo hypothetical document bằng LLM, embed nó và truy vấn.

    Nếu không sinh được hypothetical document (thiếu key / lỗi LLM) thì
    fallback về semantic_search(query, top_k) — pipeline không bao giờ crash.
    """
    hypo = hypothetical_document(query)
    if not hypo:
        return semantic_search(query, top_k=top_k)

    model = get_embedding_model()
    query_vector = model.encode(hypo).tolist()
    results = _search_with_embedding(query_vector, top_k)
    print(f"  ↳ HyDE: đã truy vấn bằng hypothetical document ({len(hypo)} chars)")
    return results


if __name__ == "__main__":
    # Test
    print("=== Semantic Search ===")
    results = semantic_search("what is the tuition fee", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
