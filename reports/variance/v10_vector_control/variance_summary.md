# Recall Lab variance, relocation_chain

4 runs of the both-agents trial.
Same scenario, same code. The spread comes from the agents answering above temperature zero. The judge and classifier run at temperature zero.

## Recall accuracy per run

| Run | sliding_window_2 | vector_topk_5 | recall_lab_brief_window_2 |
|---|---|---|---|
| run 1 | 0.00 | 0.80 | 1.00 |
| run 2 | 0.00 | 0.40 | 1.00 |
| run 3 | 0.00 | 0.60 | 1.00 |
| run 4 | 0.00 | 0.40 | 1.00 |

## Spread

- sliding_window_2: min 0.00, max 0.00, mean 0.00
- vector_topk_5: min 0.40, max 0.80, mean 0.55
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
- 2/4  Which city did I ship to right before my current one?
- 0/4  What was the very first city I shipped to?
- 4/4  What is my favorite color?
- 4/4  What food restriction should you remember for my daughter?

### recall_lab_brief_window_2

- 4/4  Where should you ship my order now?
- 4/4  Which city did I ship to right before my current one?
- 4/4  What was the very first city I shipped to?
- 4/4  What is my favorite color?
- 4/4  What food restriction should you remember for my daughter?

## Judge audit

The judge ran 3 calls per answer. 0 of 60 graded answers got a split verdict.

Every answer was unanimous. One judge call is enough.
