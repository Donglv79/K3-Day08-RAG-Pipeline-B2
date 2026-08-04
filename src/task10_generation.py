"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"

Gợi ý LLM: OpenRouter có nhiều model gắn hậu tố ":free" không tính phí — xem
https://openrouter.ai/models?max_price=0 — phù hợp nếu chưa có credit trả phí.
Base URL: "https://openrouter.ai/api/v1", dùng chung interface với OpenAI SDK.
"""

import os
from dotenv import load_dotenv

load_dotenv()

from .task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: đủ diverse nhưng không quá random
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần factual, ít sáng tạo
TEMPERATURE = 0.3

# Gemini 3.5 Flash-Lite is the selected provider for the evaluation run.
LLM_MODEL = "gemini-3.5-flash-lite"

# Keep the evaluation deterministic: use the configured Gemini provider directly
# instead of silently falling through to a different provider/model.
LLM_PROVIDERS = [
    {"name": "gemini", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
     "api_key_env": "RAG_GEMINI_API_KEY", "model": LLM_MODEL},
]


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Bạn là trợ lý trả lời câu hỏi về dịch vụ và chính sách đại học
(học phí, học bổng, ký túc xá, thư viện, đăng ký học phần).

Quy tắc bắt buộc:
1. Chỉ sử dụng thông tin từ context được cung cấp — KHÔNG bịa đặt
2. Mỗi khẳng định phải có trích dẫn ngay sau, ví dụ: [Tuition Fees, 2026]
3. Nếu context không đủ thông tin → trả lời: "Tôi không thể xác minh thông tin này từ nguồn hiện có"
4. Trả lời bằng tiếng Việt, có cấu trúc rõ ràng theo đoạn văn
5. Không suy luận hay mở rộng ngoài những gì được nêu trong context"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    (best first, worst in middle, second-best last)

    Args:
        chunks: List sorted by score descending (from retrieval)

    Returns:
        List reordered để maximize LLM attention.
    """
    if len(chunks) <= 2:
        return chunks

    front = chunks[::2]   # index 0, 2, 4 -> đặt ở đầu
    back = chunks[1::2]   # index 1, 3    -> đặt ở cuối (reversed)
    return front + back[::-1]


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("metadata", {}).get("source", f"Source {i}")
        doc_type = chunk.get("metadata", {}).get("type", "unknown")
        context_parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type}]\n"
            f"{chunk['content']}\n"
        )
    return "\n---\n".join(context_parts)


# =============================================================================
# LLM CALL — với fallback chain (OpenRouter -> OpenAI -> Gemini)
# =============================================================================

def _call_llm(messages: list[dict]) -> str:
    """
    Gọi LLM qua chain provider, thử lần lượt cho tới khi có provider thành công.

    Lý do cần fallback: OpenRouter free tier giới hạn ~50 req/ngày, dễ gặp 429
    khi cả lớp cùng demo. OPENAI_API_KEY/GEMINI_API_KEY là tùy chọn — provider
    nào không có key trong .env sẽ bị bỏ qua, không raise lỗi.
    """
    from openai import OpenAI

    last_error = None
    tried_any_key = False

    for provider in LLM_PROVIDERS:
        api_key = os.getenv(provider["api_key_env"])
        if not api_key:
            continue
        tried_any_key = True

        try:
            client = OpenAI(api_key=api_key, base_url=provider["base_url"])
            request = {"model": provider["model"], "messages": messages}
            # Gemini 3.5 Flash-Lite deprecates sampling parameters; omit them
            # rather than relying on parameters that may become HTTP 400 errors.
            if provider["name"] != "gemini":
                request.update({"temperature": TEMPERATURE, "top_p": TOP_P})
            response = client.chat.completions.create(**request)
            return response.choices[0].message.content
        except Exception as e:
            last_error = e
            continue

    if not tried_any_key:
        raise RuntimeError(
            "Thiếu API key cho LLM. Hãy điền OPENROUTER_API_KEY "
            "(hoặc OPENAI_API_KEY/GEMINI_API_KEY) vào file .env."
        )
    raise RuntimeError(f"Tất cả LLM provider đều lỗi. Lỗi cuối: {last_error}")


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(
    query: str,
    top_k: int = TOP_K,
    use_reranking: bool = True,
) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM
        6. Return answer + sources

    Args:
        query: Câu hỏi của user
        top_k: Số chunks tối đa đưa vào generation.
        use_reranking: Chuyển tiếp sang Task 9 để chạy evaluation A/B.

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    # Step 1: Retrieve
    chunks = retrieve(query, top_k=top_k, use_reranking=use_reranking)

    # Không có evidence -> không gọi LLM đoán, trả thẳng câu từ chối
    if not chunks:
        return {
            "answer": "I cannot verify this information",
            "sources": [],
            "retrieval_source": "none",
        }

    # Step 2: Reorder để tránh lost-in-the-middle
    reordered = reorder_for_llm(chunks)

    # Step 3: Format context với source labels
    context = format_context(reordered)

    # Step 4: Build prompt
    user_message = f"""Context:\n{context}\n\n---\n\nQuestion: {query}"""

    # Step 5: Call LLM (fallback chain OpenRouter -> OpenAI -> Gemini)
    answer = _call_llm([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ])

    # Step 6: Return
    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid"),
    }


if __name__ == "__main__":
    test_queries = [
        "Học phí tại RMIT Vietnam là bao nhiêu?",
        "Làm sao để đặt phòng học nhóm ở thư viện?",
        "Sinh viên quốc tế có những học bổng nào?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
