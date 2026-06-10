# Recall Lab variance, relocation_chain

5 runs of the both-agents trial.
Same scenario, same code. The spread comes from the agents answering above temperature zero. The judge and classifier run at temperature zero.

## Recall accuracy per run

| Run | recall_lab_brief_window_2 | budgeted_window |
|---|---|---|
| run 1 | 1.00 | 0.00 |
| run 2 | 1.00 | 0.00 |
| run 3 | 1.00 | 0.20 |
| run 4 | 1.00 | 0.00 |
| run 5 | 1.00 | 0.00 |

## Spread

- recall_lab_brief_window_2: min 1.00, max 1.00, mean 1.00
- budgeted_window: min 0.00, max 0.20, mean 0.04

## Per-question pass count across runs

### recall_lab_brief_window_2

- 5/5  Where should you ship my order now?
- 5/5  Which city did I ship to right before my current one?
- 5/5  What was the very first city I shipped to?
- 5/5  What is my favorite color?
- 5/5  What food restriction should you remember for my daughter?

### budgeted_window

- 0/5  Where should you ship my order now?
- 0/5  Which city did I ship to right before my current one?
- 0/5  What was the very first city I shipped to?
- 0/5  What is my favorite color?
- 1/5  What food restriction should you remember for my daughter?

## Judge audit

The judge ran 3 calls per answer. 0 of 50 graded answers got a split verdict.

Every answer was unanimous. One judge call is enough.
