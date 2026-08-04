"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API

Lưu ý: API `/retrieval` của PageIndex hiện đã deprecated (vẫn hoạt động, nhưng response
có field "deprecation" cảnh báo) và trả kết quả trong "retrieved_nodes" — mỗi node có
"relevant_contents": list[list[{section_title, relevant_content}]]. In response thật ra
(json.dumps(...)) trước khi viết logic parse, đừng đoán schema từ ví dụ code cũ.
"""

import os
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
PAGEINDEX_DOCUMENT_ID = os.getenv("PAGEINDEX_DOCUMENT_ID", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    # TODO: Implement upload
    #
    # Tham khảo: https://github.com/VectifyAI/PageIndex
    #
    # from pageindex.client import PageIndexClient
    #
    # client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    #
    # for md_file in STANDARDIZED_DIR.rglob("*.md"):
    #     # Lưu ý: PageIndex nhận PDF, không nhận .md trực tiếp — có thể cần
    #     # convert markdown sang PDF đơn giản bằng fpdf2 trước khi upload.
    #     resp = client.submit_document(str(pdf_path))
    #     doc_id = resp.get("doc_id") or resp.get("id")
    #     print(f"  ✓ Uploaded: {md_file.name} -> {doc_id}")
    if not PAGEINDEX_API_KEY:
        print("⚠ Thiếu PAGEINDEX_API_KEY; bỏ qua upload.")
        return []
    try:
        from pageindex.client import PageIndexClient
    except ImportError:
        print("⚠ Chưa cài pageindex; bỏ qua upload.")
        return []

    # PageIndex nhận tài liệu theo file. Markdown cần được nhóm chuyển sang PDF
    # trước khi upload; hàm này chỉ upload PDF đã chuẩn bị ở PAGEINDEX_UPLOAD_DIR.
    upload_dir = Path(os.getenv("PAGEINDEX_UPLOAD_DIR", "data/pageindex_pdfs"))
    if not upload_dir.exists():
        print(f"⚠ Không tìm thấy thư mục PDF upload: {upload_dir}")
        return []
    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    document_ids = []
    for pdf_file in upload_dir.glob("*.pdf"):
        response = client.submit_document(str(pdf_file))
        document_id = response.get("doc_id") or response.get("id")
        if document_id:
            document_ids.append(str(document_id))
    return document_ids


def _parse_retrieval(retrieval: dict, top_k: int) -> list[dict]:
    """Parse retrieved_nodes/relevant_contents theo response PageIndex hiện hành."""
    results = []
    for node in retrieval.get("retrieved_nodes", []):
        for group in node.get("relevant_contents", []):
            for item in group:
                content = item.get("relevant_content", "").strip()
                if content:
                    results.append({
                        "content": content,
                        "score": round(1.0 / (len(results) + 1), 4),
                        "metadata": {"section": item.get("section_title", "Unknown")},
                        "source": "pageindex",
                    })
                    if len(results) >= top_k:
                        return results
    return results


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    if top_k <= 0 or not query.strip() or not PAGEINDEX_API_KEY or not PAGEINDEX_DOCUMENT_ID:
        return []
    try:
        from pageindex.client import PageIndexClient

        client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
        response = client.submit_query(doc_id=PAGEINDEX_DOCUMENT_ID, query=query)
        retrieval_id = response.get("retrieval_id") or response.get("id")
        if not retrieval_id:
            return []
        # API chạy bất đồng bộ; poll ngắn, hết thời gian thì fallback empty an toàn.
        for _ in range(12):
            retrieval = client.get_retrieval(retrieval_id)
            if retrieval.get("status") == "completed":
                return _parse_retrieval(retrieval, top_k)
            if retrieval.get("status") in {"failed", "error"}:
                return []
            time.sleep(1)
    except Exception as exc:
        print(f"⚠ PageIndex fallback không khả dụng ({exc.__class__.__name__})")
    return []


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("tuition fee payment methods", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
