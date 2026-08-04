# RAG Evaluation Results

## Framework and setup

- Framework: RAGAS
- Dataset: 15 evidence-backed questions
- Config A: `hybrid_rerank`
- Config B: `hybrid_no_rerank`
- Evaluator: 9Router model `9router`; embeddings: local `BAAI/bge-m3`

## Overall scores

| Metric | hybrid_rerank | hybrid_no_rerank | Delta (A-B) |
|---|---:|---:|---:|
| faithfulness | 0.856 | 0.911 | -0.055 |
| answer_relevancy | 0.675 | 0.666 | +0.010 |
| context_recall | 0.733 | 0.733 | +0.000 |
| context_precision | 0.510 | 0.567 | -0.057 |
| **Average** | **0.693** | **0.719** | **-0.026** |

## A/B conclusion

`hybrid_no_rerank` is the winner by mean of the four RAGAS metrics (0.719). This conclusion is based on this fixed 15-question dataset; rerun it after changing corpus, chunking, retrieval, or model settings.

## Worst performers (bottom 3)

| Config | Question | Average metric | Root cause |
|---|---|---:|---|
| hybrid_no_rerank | Một sinh viên có thể nhận đồng thời bao nhiêu học bổng học phí? | 0.083 | Answer is not sufficiently focused on the question |
| hybrid_rerank | Sinh viên mới có thể bảo lưu học bổng không? | 0.125 | Answer is not sufficiently focused on the question |
| hybrid_rerank | Tôi có thể đổi học bổng RMIT thành tiền mặt hoặc chuyển cho người khác không? | 0.125 | Answer is not sufficiently focused on the question |

## Recommendations

1. Inspect the bottom-three queries and add or correct source chunks before changing prompts.
2. If context precision is weak, reduce candidate noise with tighter chunking or stronger reranking.
3. If faithfulness is weak, require citation for every factual claim and refuse answers with insufficient evidence.
