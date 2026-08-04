"""
Task 4 — Chunking & Indexing vào Vector Store.

    Pipeline: load (data/standardized/*.md) → chunk → embed → index ChromaDB.

Lựa chọn kỹ thuật (giải thích):
    - Chunking: RecursiveCharacterTextSplitter (an toàn, giữ nguyên cấu trúc câu).
      CHUNK_SIZE=800  → đủ ngữ cảnh cho 1 đoạn chính sách, vừa context window LLM.
      CHUNK_OVERLAP=100 → tránh câu văn quan trọng bị cắt đôi ở ranh giới 2 chunk.
    - Embedding model: BAAI/bge-m3 (1024 dim) → multilingual, tốt cho cả tiếng
      Việt lẫn tiếng Anh (đúng khuyến nghị LAB_GUIDE CP2).
    - Vector store: ChromaDB local persistent, không cần Docker, space=cosine.

Lưu ý: nếu đổi corpus phải XÓA chroma_db/ cũ trước khi reindex (tránh trộn dữ liệu).
"""

from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"


# =============================================================================
# CONFIGURATION
# =============================================================================

# RecursiveCharacterTextSplitter: cắt theo thứ tự ưu tiên \n\n → \n → câu → từ.
# Size 800 ký tự ~ 1 đoạn văn chính sách; Overlap 100 ký tự giữ nguyên vẹn câu
# quan trọng nằm ở biên giới giữa 2 chunk (đúng tham số LAB_GUIDE).
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# BAAI/bge-m3: model embedding đa ngôn ngữ (tốt tiếng Việt), vector 1024 chiều.
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# ChromaDB local persistent, dùng metric cosine để query đúng khoảng cách cosine.
VECTOR_STORE = "chromadb"
COLLECTION_NAME = "university_services_docs"


# =============================================================================
# SINGLETON HELPERS — dùng chung cho Task 5 (tránh nạp lại model mỗi lần query)
# =============================================================================

_embedding_model = None
_collection = None


def get_embedding_model():
    """Lazy-load model embedding (singleton)."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        print(f"  Loading embedding model: {EMBEDDING_MODEL} ...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def get_collection():
    """Lazy-open ChromaDB collection (singleton)."""
    global _collection
    if _collection is None:
        import chromadb

        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in str(md_file) else "news"
        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type},
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i},
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    model = get_embedding_model()
    texts = [c["content"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn (ChromaDB).
    """
    collection = get_collection()

    ids = [
        f"{c['metadata']['source']}_chunk_{c['metadata']['chunk_index']}"
        for c in chunks
    ]
    collection.upsert(
        ids=ids,
        documents=[c["content"] for c in chunks],
        embeddings=[c["embedding"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )


def clear_vectorstore():
    """Xoá toàn bộ data cũ trong collection (tránh trộn dữ liệu khi reindex)."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
        print("✓ Đã xoá collection cũ")
    except Exception:
        print("  (Chưa có collection cũ để xoá)")


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")

    collection = get_collection()
    print(f"  Collection '{COLLECTION_NAME}' có {collection.count()} chunks")


if __name__ == "__main__":
    clear_vectorstore()
    run_pipeline()
