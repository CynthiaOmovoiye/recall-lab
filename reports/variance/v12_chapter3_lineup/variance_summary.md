# Recall Lab variance, relocation_chain

5 runs of the both-agents trial.
Same scenario, same code. The spread comes from the agents answering above temperature zero. The judge and classifier run at temperature zero.

## Recall accuracy per run

| Run | sliding_window_2 | vector_topk_5 | strong_rag | episodic_judge | recall_lab_brief_window_2 |
|---|---|---|---|---|---|
| run 1 | 0.00 | 0.60 | 0.60 | 1.00 | 1.00 |
| run 2 | 0.00 | 0.40 | 0.80 | 1.00 | 1.00 |
| run 3 | 0.00 | 0.60 | 0.80 | 1.00 | 1.00 |
| run 4 | 0.00 | 0.40 | 0.80 | 1.00 | 1.00 |
| run 5 | 0.00 | 0.60 | 0.80 | 1.00 | 1.00 |

## Spread

- sliding_window_2: min 0.00, max 0.00, mean 0.00
- vector_topk_5: min 0.40, max 0.60, mean 0.52
- strong_rag: min 0.60, max 0.80, mean 0.76
- episodic_judge: min 1.00, max 1.00, mean 1.00
- recall_lab_brief_window_2: min 1.00, max 1.00, mean 1.00

## Per-question pass count across runs

### sliding_window_2

- 0/5  Where should you ship my order now?
- 0/5  Which city did I ship to right before my current one?
- 0/5  What was the very first city I shipped to?
- 0/5  What is my favorite color?
- 0/5  What food restriction should you remember for my daughter?

### vector_topk_5

- 0/5  Where should you ship my order now?
- 3/5  Which city did I ship to right before my current one?
- 0/5  What was the very first city I shipped to?
- 5/5  What is my favorite color?
- 5/5  What food restriction should you remember for my daughter?

### strong_rag

- 4/5  Where should you ship my order now?
- 5/5  Which city did I ship to right before my current one?
- 0/5  What was the very first city I shipped to?
- 5/5  What is my favorite color?
- 5/5  What food restriction should you remember for my daughter?

### episodic_judge

- 5/5  Where should you ship my order now?
- 5/5  Which city did I ship to right before my current one?
- 5/5  What was the very first city I shipped to?
- 5/5  What is my favorite color?
- 5/5  What food restriction should you remember for my daughter?

### recall_lab_brief_window_2

- 5/5  Where should you ship my order now?
- 5/5  Which city did I ship to right before my current one?
- 5/5  What was the very first city I shipped to?
- 5/5  What is my favorite color?
- 5/5  What food restriction should you remember for my daughter?

## Judge audit

The judge ran 3 calls per answer. 0 of 125 graded answers got a split verdict.

Every answer was unanimous. One judge call is enough.
