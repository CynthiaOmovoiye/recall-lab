# Recall Lab variance, relocation_chain

4 runs of the both-agents trial.
Same scenario, same code. The spread comes from the agents answering above temperature zero. The judge and classifier run at temperature zero.

## Recall accuracy per run

| Run | sliding_window_2 | vector_topk_5 | strong_rag | strong_rag_dated | episodic_judge | recall_lab_brief_window_2 |
|---|---|---|---|---|---|---|
| run 1 | 0.00 | 0.40 | 0.80 | 0.40 | 1.00 | 1.00 |
| run 2 | 0.00 | 0.40 | 0.80 | 0.40 | 1.00 | 1.00 |
| run 3 | 0.00 | 0.80 | 0.80 | 0.40 | 1.00 | 1.00 |
| run 4 | 0.00 | 0.40 | 0.80 | 0.40 | 1.00 | 1.00 |

## Spread

- sliding_window_2: min 0.00, max 0.00, mean 0.00
- vector_topk_5: min 0.40, max 0.80, mean 0.50
- strong_rag: min 0.80, max 0.80, mean 0.80
- strong_rag_dated: min 0.40, max 0.40, mean 0.40
- episodic_judge: min 1.00, max 1.00, mean 1.00
- recall_lab_brief_window_2: min 1.00, max 1.00, mean 1.00

## Per-question pass count across runs

### sliding_window_2

- 0/4  Where should you ship my order now?
- 0/4  Which city did I ship to right before my current one?
- 0/4  What was the very first city I shipped to?
- 0/4  What is my favorite color?
- 0/4  What food restriction should you remember for my daughter?

### vector_topk_5

- 1/4  Where should you ship my order now?
- 1/4  Which city did I ship to right before my current one?
- 0/4  What was the very first city I shipped to?
- 4/4  What is my favorite color?
- 4/4  What food restriction should you remember for my daughter?

### strong_rag

- 4/4  Where should you ship my order now?
- 4/4  Which city did I ship to right before my current one?
- 0/4  What was the very first city I shipped to?
- 4/4  What is my favorite color?
- 4/4  What food restriction should you remember for my daughter?

### strong_rag_dated

- 4/4  Where should you ship my order now?
- 0/4  Which city did I ship to right before my current one?
- 0/4  What was the very first city I shipped to?
- 4/4  What is my favorite color?
- 0/4  What food restriction should you remember for my daughter?

### episodic_judge

- 4/4  Where should you ship my order now?
- 4/4  Which city did I ship to right before my current one?
- 4/4  What was the very first city I shipped to?
- 4/4  What is my favorite color?
- 4/4  What food restriction should you remember for my daughter?

### recall_lab_brief_window_2

- 4/4  Where should you ship my order now?
- 4/4  Which city did I ship to right before my current one?
- 4/4  What was the very first city I shipped to?
- 4/4  What is my favorite color?
- 4/4  What food restriction should you remember for my daughter?

## Judge audit

The judge ran 3 calls per answer. 1 of 120 graded answers got a split verdict.

Split verdicts:
- run 3, vector_topk_5: "Which city did I ship to right before my current one?" votes: hallucinated, correct, correct
