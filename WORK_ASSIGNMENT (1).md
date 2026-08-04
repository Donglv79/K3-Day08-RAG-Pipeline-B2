# Phân Công Nhóm — University Services RAG

## Phương án B — Nhóm 5 thành viên, chuyên sâu Retrieval

Mục tiêu: mỗi người phụ trách một lớp rõ ràng của hệ thống, trong đó retrieval được tách thành dense retrieval và sparse/reranking/fallback để dễ tối ưu và đánh giá.

## Phân công chính thức

| Thành viên | MSSV | Vai trò | Task owner | File phụ trách | Deliverable chính | Trạng thái |
|---|---|---|---|---|---|---|
| [Tên thành viên 1] | [MSSV] | Team Leader & RAG Architect | Task 9 | `src/task9_retrieval_pipeline.py`; `src/supervisor.py` nếu bổ sung | Điều phối, ghép retrieval pipeline, kiểm tra interface giữa các module | Chưa bắt đầu |
| [Tên thành viên 2] | [MSSV] | Data & Dense Search Dev | Task 1–5 | `src/task1_collect_legal_docs.py` đến `src/task5_semantic_search.py` | Data, Markdown, ChromaDB, semantic search; HyDE là bonus | Chưa bắt đầu |
| [Tên thành viên 3] | [MSSV] | Sparse Search & Advanced Reranking Dev | Task 6–8 | `src/task6_lexical_search.py`, `src/task7_reranking.py`, `src/task8_pageindex_vectorless.py` | BM25/TF-IDF, RRF reranking, PageIndex fallback | Chưa bắt đầu |
| [Tên thành viên 4] | [MSSV] | Frontend & Chatbot Developer | Task 10 | `src/task10_generation.py`, `app.py` | Generation có citation, Streamlit chatbot, source display và chat history | Chưa bắt đầu |
| [Tên thành viên 5] | [MSSV] | Evaluation & QA Engineer | Evaluation nhóm | `group_project/evaluation/golden_dataset.json`, `eval_pipeline.py`, `results.md` | Golden dataset ≥15 câu, RAGAS, so sánh A/B và regression QA | Chưa bắt đầu |

## Phạm vi từng vai trò

### 1. Team Leader & RAG Architect

- Điều phối tiến độ, nhánh code, merge và thống nhất schema giữa các module.
- Implement `retrieve()` (Task 9): dense + sparse → RRF → rerank → PageIndex fallback.
- So sánh `score_threshold` với **cosine score gốc** từ `semantic_search`, không dùng RRF score.
- Starter hiện không có `src/supervisor.py`; chỉ tạo file này nếu nhóm muốn bổ sung orchestration. Nó không thay thế Task 9.
- Phối hợp QA Engineer chạy regression sau mỗi lần tích hợp.

### 2. Data & Dense Search Dev

- Task 1: thu thập ≥3 PDF/DOCX vào `data/landing/legal/`.
- Task 2: crawl/tạo ≥5 JSON trong `data/landing/news/`, gồm `url`, `title`, `date_crawled`, `content_markdown`.
- Task 3: convert toàn bộ dữ liệu thành Markdown ở `data/standardized/`.
- Task 4: chunk, embed và index ChromaDB collection `university_services_docs`; comment rõ chunk size, overlap, model và dimension.
- Task 5: implement `semantic_search(query, top_k)` trả kết quả có cosine score giảm dần.
- Bonus: bổ sung HyDE để tạo hypothetical document trước khi vector hóa query.

### 3. Sparse Search & Advanced Reranking Dev

- Task 6: implement `lexical_search(query, top_k)` bằng BM25; TF-IDF là bonus.
- Task 7: implement `rerank_rrf()` và `rerank()` theo interface dùng được trong Task 9.
- Task 8: implement `pageindex_search()` và trả kết quả có `source: "pageindex"`.
- Khi thiếu `PAGEINDEX_API_KEY`, PageIndex phải không làm crash pipeline; trả list rỗng hoặc xử lý fallback an toàn.
- Duy trì schema chung:

```python
{"content": str, "score": float, "metadata": dict}
```

### 4. Frontend & Chatbot Developer

- Task 10: implement reordering, `format_context()` và `generate_with_citation()`.
- Câu trả lời phải có citation; thiếu evidence phải trả `I cannot verify this information`.
- Kết nối `app.py` với Task 10, hiển thị answer, chunks nguồn và lịch sử chat.
- Phối hợp Leader kiểm thử luồng từ câu hỏi tới câu trả lời trên giao diện.

### 5. Evaluation & QA Engineer

- Bổ sung ít nhất 12 câu để `golden_dataset.json` đạt ≥15 cặp Q&A có expected answer/context.
- Implement RAGAS trong `eval_pipeline.py`: faithfulness, answer relevance, context recall, context precision.
- Chạy A/B: khuyến nghị A = hybrid + rerank; B = dense-only hoặc hybrid không rerank.
- Hoàn thành `results.md`: điểm số, worst performers và đề xuất cải tiến.
- Chạy full pytest, ghi nhận lỗi integration và chuyển về đúng owner sửa.

## Luồng tích hợp

```text
Data & Dense Search Dev ────────────────┐
Sparse Search & Advanced Reranking Dev ─┼→ Team Leader & RAG Architect (Task 9)
                                        ├→ Frontend & Chatbot Dev (Task 10 + app.py)
                                        └→ Evaluation & QA Engineer (RAGAS + A/B)
```

## Quy ước nhóm

- Không đổi schema output của task khác khi chưa thống nhất với owner.
- Không commit API key hoặc file `.env`.
- Mỗi owner chạy pytest phần mình trước khi merge.
- Sau khi merge, QA Engineer chạy `pytest tests/test_individual.py -v`.

## Lệnh kiểm tra theo vai trò

```bash
# Team Leader & RAG Architect
pytest tests/test_individual.py::TestTask9 -v

# Data & Dense Search Dev
pytest tests/test_individual.py::TestTask1 tests/test_individual.py::TestTask2 tests/test_individual.py::TestTask3 tests/test_individual.py::TestTask4 tests/test_individual.py::TestTask5 -v

# Sparse Search & Advanced Reranking Dev
pytest tests/test_individual.py::TestTask6 tests/test_individual.py::TestTask7 tests/test_individual.py::TestTask8 -v

# Frontend & Chatbot Developer
pytest tests/test_individual.py::TestTask10 -v

# Evaluation & QA Engineer
pytest tests/test_individual.py -v
```

## Báo cáo cá nhân

Mỗi thành viên nộp report 1–2 trang: nhiệm vụ owner, file/hàm đã làm, lý do kỹ thuật, pytest/log hoặc screenshot, một lỗi và cách sửa, cùng cách phần việc nối vào pipeline chung.

## Checklist triển khai chi tiết

> Đánh dấu `[x]` khi hoàn thành. Mỗi người commit phần việc của mình trên branch riêng và chỉ merge sau khi test pass.

### Thành viên 1 — Team Leader & RAG Architect (Task 9)

**Mục tiêu:** tích hợp mọi module retrieval thành một pipeline ổn định, đúng fallback logic.

- [ ] Tạo branch `feature/task9-retrieval-pipeline` và đọc interface của Task 5–8.
- [ ] Chốt schema kết quả chung: `content`, `score`, `metadata`, `source`.
- [ ] Trong `retrieve()`, gọi `semantic_search(query, top_k * 2)` và `lexical_search(query, top_k * 2)`.
- [ ] Merge hai ranked lists bằng `rerank_rrf()`.
- [ ] Gắn `source: "hybrid"` cho kết quả đã fuse.
- [ ] Gọi `rerank()` hoặc giữ kết quả RRF tùy cờ `use_reranking`.
- [ ] Lấy `best_score` từ kết quả dense gốc, không lấy score sau RRF.
- [ ] Nếu dense score thấp hơn `score_threshold`, gọi `pageindex_search()`; chỉ dùng fallback nếu PageIndex có kết quả.
- [ ] Bảo đảm query rác hoặc PageIndex thiếu API key không làm pipeline crash.
- [ ] Chạy test Task 9 và ít nhất 4 query thủ công: học phí, học bổng, thư viện, query vô nghĩa.
- [ ] Điều phối integration: thông báo lỗi interface cho đúng owner sửa; không tự sửa file owner khác nếu chưa thống nhất.

**Bàn giao:** `retrieve()` trả list tối đa `top_k`, mỗi item có `content`, `score`, `metadata`, `source` (`hybrid` hoặc `pageindex`).

### Thành viên 2 — Data & Dense Search Dev (Task 1–5)

**Mục tiêu:** cung cấp corpus sạch, vector database và tìm kiếm ngữ nghĩa làm đầu vào cho pipeline.

#### Task 1 — Legal documents

- [ ] Tạo/kiểm tra `data/landing/legal/`.
- [ ] Tải tối thiểu 3 file PDF/DOCX từ nguồn công khai, ưu tiên học phí, học bổng, accommodation và course registration.
- [ ] Đặt tên file không dấu, mô tả rõ chủ đề, ví dụ `tuition-fees-rmit.pdf`.
- [ ] Kiểm tra từng file không rỗng và mở được.
- [ ] Ghi URL nguồn vào commit message hoặc một ghi chú trong report cá nhân.

#### Task 2 — News crawling

- [ ] Chọn tối thiểu 5 URL công khai về thư viện, hỗ trợ sinh viên, sự kiện hoặc học bổng.
- [ ] Điền `ARTICLE_URLS` và implement `crawl_article()` trong `src/task2_crawl_news.py`.
- [ ] Lưu một JSON cho mỗi bài tại `data/landing/news/`.
- [ ] Kiểm tra mỗi JSON có `url`, `title`, `date_crawled`, `content_markdown` và nội dung đủ dài.
- [ ] Nếu website chặn crawler, dùng nguồn công khai khác hoặc dữ liệu mẫu có nguồn rõ ràng; không cố vượt cơ chế bảo vệ website.

#### Task 3 — Markdown conversion

- [ ] Implement `convert_legal_docs()` dùng MarkItDown cho PDF/DOCX.
- [ ] Implement `convert_news_articles()` để chuyển JSON sang Markdown, thêm title/source/crawled date ở header.
- [ ] Giữ cấu trúc `data/standardized/legal/` và `data/standardized/news/`.
- [ ] Chạy `convert_all()` và kiểm tra Markdown có nội dung lớn hơn 200 ký tự.

#### Task 4 — ChromaDB indexing

- [ ] Implement `load_documents()` đọc toàn bộ `.md` và gắn metadata `source`, `type`.
- [ ] Implement `chunk_documents()` với `CHUNK_SIZE`, `CHUNK_OVERLAP`; ghi lý do kỹ thuật bằng comment.
- [ ] Implement `embed_chunks()` bằng model đã chọn và bảo đảm dimension khớp config.
- [ ] Implement `index_to_vectorstore()` vào collection `university_services_docs`.
- [ ] Khi reindex corpus mới, xóa/clear collection cũ có chủ đích để không lẫn chunks cũ.
- [ ] Chạy pipeline, kiểm tra `chroma_db/` được sinh và collection có documents.

#### Task 5 — Semantic search

- [ ] Load đúng Chroma collection và embedding model giống Task 4.
- [ ] Implement `semantic_search(query, top_k)`; chuyển Chroma distance thành cosine similarity.
- [ ] Trả kết quả giảm dần theo `score`, không vượt quá `top_k`.
- [ ] Kiểm tra ít nhất query tiếng Anh và tiếng Việt về học phí/học bổng.
- [ ] Bonus tùy chọn: implement HyDE, có cờ bật/tắt và mô tả trade-off độ trễ/chi phí.

**Bàn giao:** data landing + Markdown + ChromaDB hoạt động; semantic search trả `content`, `score`, `metadata`.

### Thành viên 3 — Sparse Search & Advanced Reranking Dev (Task 6–8)

**Mục tiêu:** bổ sung exact keyword retrieval, fusion/reranking và fallback cho các truy vấn không đủ evidence.

#### Task 6 — BM25 / lexical search

- [ ] Chọn corpus dùng chung với Task 4: ưu tiên chunks đã chuẩn hóa để dense và sparse trả cùng mức granularity.
- [ ] Implement `build_bm25_index(corpus)` bằng `BM25Okapi`.
- [ ] Tokenize nhất quán: lowercase, tách whitespace; có thể bổ sung xử lý dấu câu nếu cần.
- [ ] Implement `lexical_search(query, top_k)`.
- [ ] Chỉ trả kết quả BM25 score dương, giảm dần, có `content`, `score`, `metadata`.
- [ ] Test thủ công các keyword chính xác: `tuition fee`, `Academic Achievement`, `myRMIT`, `library study room`.
- [ ] Bonus tùy chọn: so sánh BM25 với TF-IDF trong report/evaluation.

#### Task 7 — RRF reranking

- [ ] Implement `rerank_rrf(ranked_lists, top_k, k=60)`.
- [ ] Deduplicate candidates theo `content` hoặc ID/source + chunk index để một chunk không lặp trong kết quả.
- [ ] Giữ metadata của candidate gốc, ghi đè `score` bằng RRF score, sort giảm dần.
- [ ] Implement `rerank(query, candidates, top_k)` sao cho chạy được với test starter; nếu dùng RRF, dùng một ranked list hoặc một lexical-overlap scorer đơn giản cho interface này.
- [ ] Ghi rõ trong comment/report: RRF score là điểm fusion theo rank, không phải cosine similarity.
- [ ] Không gọi API reranker bắt buộc nếu không có API key; có fallback local/deterministic để test luôn chạy.

#### Task 8 — PageIndex fallback

- [ ] Đọc `PAGEINDEX_API_KEY` từ `.env`; không hard-code secret.
- [ ] Lưu/cấu hình document ID sau khi upload; không upload lại không cần thiết mỗi query.
- [ ] Implement `pageindex_search(query, top_k)` và parse response theo schema thực tế `retrieved_nodes` / `relevant_contents`.
- [ ] Kết quả trả `content`, `score`, `metadata`, `source: "pageindex"`.
- [ ] Khi thiếu API key, API lỗi hoặc chưa upload tài liệu: log gọn, trả `[]`, không raise exception làm hỏng Task 9.
- [ ] Test query fallback với `score_threshold=0.99` thông qua Task 9 sau khi Leader tích hợp.

**Bàn giao:** các hàm Task 6–8 không crash, đúng schema và có commit mô tả giới hạn khi PageIndex chưa cấu hình.

### Thành viên 4 — Frontend & Chatbot Developer (Task 10)

**Mục tiêu:** tạo chatbot usable, trả lời dựa trên retrieval và hiển thị citation/source rõ ràng.

- [ ] Implement `reorder_for_llm()` theo `front + back[::-1]`, giữ nguyên số chunks và chunk tốt nhất ở đầu.
- [ ] Implement `format_context()`; mỗi chunk phải chứa source và type để LLM cite được.
- [ ] Implement `generate_with_citation()` gọi `retrieve()` trước khi gọi LLM.
- [ ] Viết system prompt yêu cầu citation dạng `[Nguồn, Năm]` và không bịa thông tin.
- [ ] Nếu không có chunks/evidence, trả answer `I cannot verify this information` thay vì gọi LLM đoán.
- [ ] Đọc API key từ `.env`; báo lỗi thân thiện nếu thiếu key.
- [ ] Hoàn thiện `app.py`: chat input, suggested questions, source expander và chat history.
- [ ] Chạy `streamlit run app.py` và test ít nhất 3 câu hỏi từ giao diện.

**Bàn giao:** `generate_with_citation()` trả `answer`, `sources`, `retrieval_source`; app chạy local không lỗi import.

### Thành viên 5 — Evaluation & QA Engineer (Evaluation nhóm)

**Mục tiêu:** chứng minh pipeline hoạt động, đo được chất lượng và phát hiện regression sớm.

#### Golden dataset

- [ ] Mở rộng `golden_dataset.json` từ 3 lên ít nhất 15 câu hỏi.
- [ ] Phủ các nhóm: tuition/payment, scholarship, accommodation, course registration, library và student support.
- [ ] Mỗi mẫu có `question`, `expected_answer`, `expected_context` dựa trên tài liệu thật.
- [ ] Không thêm expected answer không có evidence trong corpus.

#### RAGAS và A/B test

- [ ] Implement loader dataset và cơ chế gọi pipeline trên từng question.
- [ ] Implement/chạy RAGAS: faithfulness, answer relevance, context recall, context precision.
- [ ] Định nghĩa config A: hybrid retrieval + reranking.
- [ ] Định nghĩa config B: dense-only hoặc hybrid không reranking.
- [ ] Lưu metric, lỗi và query khó cho từng config.
- [ ] Hoàn thiện `results.md`: bảng điểm, winner A/B, worst performers, nguyên nhân và hướng cải thiện.

#### QA integration

- [ ] Chạy pytest riêng của owner sau mỗi merge khi có thể.
- [ ] Chạy `pytest tests/test_individual.py -v` trước demo.
- [ ] Test smoke end-to-end từ app: câu hỏi có evidence, câu hỏi lạc đề, và câu hỏi follow-up.
- [ ] Lập danh sách bug có bước tái hiện, expected/actual result và owner phụ trách.

**Bàn giao:** golden dataset ≥15, script evaluation có hướng dẫn chạy, `results.md` hoàn chỉnh và log test cuối cùng.

## Mốc bàn giao đề xuất

| Mốc | Owner chính | Điều kiện bàn giao |
|---|---|---|
| M1 — Corpus sẵn sàng | Thành viên 2 | ≥3 legal files, ≥5 news JSON, Markdown đã convert |
| M2 — Retrieval độc lập | Thành viên 2, 3 | ChromaDB, semantic search, BM25, RRF chạy được |
| M3 — Pipeline hoàn chỉnh | Thành viên 1 | Task 9 ghép được hybrid/fallback, không crash |
| M4 — Sản phẩm demo | Thành viên 4 | Task 10 + `app.py` hiển thị answer và sources |
| M5 — Đánh giá/nộp bài | Thành viên 5 | 15+ golden Q&A, A/B, `results.md`, full pytest |

## Checkpoint theo thời gian thực hiện

> Các checkpoint dưới đây áp dụng cho **5 vai trò đã chốt** trong tài liệu này. Domain của repo là University Services/RMIT Vietnam; vì vậy dữ liệu, metadata và câu hỏi đánh giá phải bám học phí, học bổng, chỗ ở, thư viện, đăng ký học phần và hỗ trợ sinh viên — không dùng checklist Shopee hoặc metadata `customer_role`.

### CP0 — Setup môi trường & khởi tạo project (0:00–0:10)

| Vai trò | Checklist |
|---|---|
| Thành viên 1 — Leader | [ ] Kiểm tra mọi người clone được repo. [ ] Tạo repo/branch strategy chung. [ ] Tạo `.env` từ `.env.example`; chia sẻ API key qua kênh riêng, không commit secret. |
| Thành viên 2 — Data & Dense | [ ] Tạo `.venv`. [ ] Chạy `pip install -r requirements.txt`. [ ] Kiểm tra import `chromadb`, `sentence_transformers`, `markitdown`. |
| Thành viên 3 — Sparse/Reranking | [ ] Kiểm tra import `rank_bm25`, `numpy`, `dotenv`. [ ] Kiểm tra `.env` đọc được nhưng không in API key ra log. |
| Thành viên 4 — Frontend | [ ] Kiểm tra `streamlit --version`. [ ] Chạy thử `streamlit run app.py` và xác nhận UI mở được, dù Task 10 chưa implement. |
| Thành viên 5 — Evaluation/QA | [ ] Kiểm tra import `ragas`, `datasets`. [ ] Kiểm tra mở được `golden_dataset.json`, hiện có 3 mẫu để mở rộng sau. |

**CP0 Passed:** mọi người cài được dependencies cần cho phần mình; không còn lỗi import thiết yếu; `.env` không bị Git theo dõi.

### CP1 — Thu thập & chuẩn hóa dữ liệu, Task 1–3 (0:10–0:35)

| Vai trò | Checklist |
|---|---|
| Thành viên 1 — Leader | [ ] Chốt danh sách chủ đề và URL nguồn với Thành viên 2. [ ] Kiểm tra không có tài liệu trùng hoặc nguồn không rõ ràng. |
| Thành viên 2 — Data & Dense | [ ] Hoàn thành Task 1: ≥3 PDF/DOCX trong `data/landing/legal/`. [ ] Hoàn thành Task 2: ≥5 news JSON trong `data/landing/news/`. [ ] Ghi đủ metadata JSON. [ ] Hoàn thành Task 3: Markdown tương ứng trong `data/standardized/`. |
| Thành viên 3 — Sparse/Reranking | [ ] Review nhanh Markdown để bảo đảm nội dung có keyword cần cho BM25: tuition, scholarship, library, accommodation, myRMIT. [ ] Báo sớm file rỗng/lỗi encoding cho Thành viên 2. |
| Thành viên 4 — Frontend | [ ] Cập nhật/kiểm tra suggested questions trong `app.py` bám đúng tài liệu đã thu thập. [ ] Chuẩn bị mock UI không phụ thuộc API key. |
| Thành viên 5 — Evaluation/QA | [ ] Lập khung 15 câu golden Q&A dựa trên nguồn được chọn. [ ] Kiểm tra số file và nội dung `.md` bằng checklist hoặc test Task 1–3. |

**CP1 Passed:** ≥3 legal files, ≥5 news files và Markdown có nội dung trong `data/standardized/`; pytest Task 1–3 pass.

### CP2 — Chunking, indexing & search cơ bản, Task 4–6 (0:35–1:00)

| Vai trò | Checklist |
|---|---|
| Thành viên 1 — Leader | [ ] Review config chunking/model trước khi index. [ ] Chốt contract kết quả search: `content`, `score`, `metadata`. |
| Thành viên 2 — Data & Dense | [ ] Hoàn thành Task 4: load, chunk, embed, index ChromaDB. [ ] Hoàn thành Task 5: cosine semantic search sorted giảm dần. [ ] Ghi comment lý do chọn chunk size/overlap/model. |
| Thành viên 3 — Sparse/Reranking | [ ] Hoàn thành Task 6: build BM25 từ corpus chuẩn hóa/chunks. [ ] `lexical_search()` trả kết quả giảm dần với score dương khi keyword khớp. |
| Thành viên 4 — Frontend | [ ] Chuẩn bị hiển thị metadata `source`, `type`, score trong source expander. [ ] Không hard-code nội dung câu trả lời trong UI. |
| Thành viên 5 — Evaluation/QA | [ ] Tạo nhóm smoke queries: relevant, exact-keyword, Vietnamese, English. [ ] Ghi điểm dense/BM25 để phục vụ calibrate threshold sau này. |

**CP2 Passed:** có `chroma_db/` với collection có dữ liệu; pytest Task 4, 5, 6 pass; cả semantic và BM25 tìm được query mẫu phù hợp.

### CP3 — Reranking & PageIndex fallback, Task 7–8 (1:00–1:20)

| Vai trò | Checklist |
|---|---|
| Thành viên 1 — Leader | [ ] Review RRF với `k=60`. [ ] Nhắc rõ RRF score chỉ dùng fusion/ranking, không dùng để đánh giá ngưỡng fallback. |
| Thành viên 2 — Data & Dense | [ ] Cung cấp ví dụ dense results và giải thích thang cosine score để Leader/QA calibrate. |
| Thành viên 3 — Sparse/Reranking | [ ] Hoàn thành Task 7: RRF merge + deduplicate + sort. [ ] Hoàn thành Task 8: PageIndex query/parse response. [ ] Bảo đảm thiếu key/API lỗi trả `[]`, không crash. |
| Thành viên 4 — Frontend | [ ] Chuẩn bị cách hiển thị `retrieval_source` là `hybrid` hoặc `pageindex` nếu cần demo. |
| Thành viên 5 — Evaluation/QA | [ ] Chạy query ngoài domain và query tổng hợp. [ ] Ghi nhận dense score để đề xuất threshold; kiểm tra fallback được gọi khi score thấp. |

**CP3 Passed:** RRF gộp được dense + BM25; PageIndex trả kết quả hoặc graceful empty fallback; pytest Task 7–8 pass.

### CP4 — Pipeline hoàn chỉnh & generation, Task 9–10 (1:20–1:45)

| Vai trò | Checklist |
|---|---|
| Thành viên 1 — Leader | [ ] Hoàn thành Task 9. [ ] Dùng `dense_results[0]["score"] < score_threshold` cho fallback. [ ] Test relevant query, query rác, PageIndex unavailable. |
| Thành viên 2 — Data & Dense | [ ] Hỗ trợ debug ChromaDB/model nếu Task 9 không lấy được dense results. [ ] Không đổi dữ liệu/index giữa lúc integration nếu chưa báo Leader. |
| Thành viên 3 — Sparse/Reranking | [ ] Hỗ trợ debug format của BM25/RRF/PageIndex. [ ] Xác nhận kết quả fused có metadata để Task 10 cite source. |
| Thành viên 4 — Frontend | [ ] Hoàn thành Task 10: reorder `front + back[::-1]`, format context, generation có citation. [ ] Khi thiếu evidence trả `I cannot verify this information`. |
| Thành viên 5 — Evaluation/QA | [ ] Review answer có citation/source. [ ] Chạy pytest Task 9–10 và regression các task trước. |

**CP4 Passed:** pipeline end-to-end không crash; citation và source xuất hiện; toàn bộ pytest `tests/test_individual.py` pass khi tất cả task đã hoàn thành.

### CP5 — Sản phẩm nhóm: Chatbot & RAGAS evaluation (1:45–2:15)

| Vai trò | Checklist |
|---|---|
| Thành viên 1 — Leader | [ ] Điều phối merge bản pipeline ổn định vào nhánh demo. [ ] Chốt config A/B cho evaluation. |
| Thành viên 2 — Data & Dense | [ ] Freeze corpus/chroma version cho lần evaluation chính thức. [ ] Hỗ trợ giải thích chênh lệch dense-only với hybrid. |
| Thành viên 3 — Sparse/Reranking | [ ] Cung cấp cờ/cấu hình tắt reranking hoặc sparse search để QA chạy A/B công bằng. |
| Thành viên 4 — Frontend | [ ] Hoàn thiện app: top_k, suggested questions, source expander, chat history và thông báo lỗi thân thiện. [ ] Test demo UI ít nhất 3 câu. |
| Thành viên 5 — Evaluation/QA | [ ] Hoàn thành ≥15 golden Q&A. [ ] Chạy 4 metrics RAGAS. [ ] So sánh A/B và viết `results.md` gồm score, worst cases, recommendations. |

**CP5 Passed:** chatbot trả lời có nguồn; `results.md` có bảng metric A/B; golden dataset có ít nhất 15 mẫu hợp lệ.

### CP6 — Demo & nộp bài (2:15–3:00)

| Vai trò | Checklist |
|---|---|
| Thành viên 1 — Leader | [ ] Trình bày kiến trúc tổng thể, retrieval flow và fallback logic. [ ] Xác nhận repo chung là phiên bản cuối. |
| Thành viên 2 — Data & Dense | [ ] Giải thích nguồn dữ liệu, chuẩn hóa, chunking, embedding, ChromaDB và semantic search. |
| Thành viên 3 — Sparse/Reranking | [ ] Giải thích BM25, vì sao dùng hybrid, công thức RRF và giới hạn/điều kiện PageIndex fallback. |
| Thành viên 4 — Frontend | [ ] Live demo Streamlit: câu hỏi, answer, citation, source documents và xử lý câu hỏi thiếu evidence. |
| Thành viên 5 — Evaluation/QA | [ ] Trình bày golden dataset, RAGAS metrics, A/B conclusion và worst performers. |

**CP6 Passed:** demo chạy được, code/data không chứa secrets đã push lên repo nhóm, report cá nhân và evaluation report sẵn sàng nộp.
