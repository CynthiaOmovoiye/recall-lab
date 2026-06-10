# Recall Lab variance, relocation_chain

5 runs of the both-agents trial.
Same scenario, same code. The spread comes from the agents answering above temperature zero. The judge and classifier run at temperature zero.

## Recall accuracy per run

| Run | sliding_window_2 | recall_lab_brief_window_2 |
|---|---|---|
| run 1 | 0.00 | 1.00 |
| run 2 | 0.00 | 0.80 |
| run 3 | 0.00 | 1.00 |
| run 4 | 0.00 | 1.00 |
| run 5 | 0.00 | 0.80 |

## Spread

- sliding_window_2: min 0.00, max 0.00, mean 0.00
- recall_lab_brief_window_2: min 0.80, max 1.00, mean 0.92

## Per-question pass count across runs

### sliding_window_2

- 0/5  Where should you ship my order now?
- 0/5  Which city did I ship to right before my current one?
- 0/5  What was the very first city I shipped to?
- 0/5  What is my favorite color?
- 0/5  What food restriction should you remember for my daughter?

### recall_lab_brief_window_2

- 5/5  Where should you ship my order now?
- 5/5  Which city did I ship to right before my current one?
- 3/5  What was the very first city I shipped to?
- 5/5  What is my favorite color?
- 5/5  What food restriction should you remember for my daughter?

## Judge audit

The judge ran 3 calls per answer. 0 of 50 graded answers got a split verdict.

Every answer was unanimous. One judge call is enough.
