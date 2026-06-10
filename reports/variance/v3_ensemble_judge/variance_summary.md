# Recall Lab variance, retail_memory_week

5 runs of the four-day both-agents trial.
Same scenario, same code. The spread comes from the judge and the classifier running above temperature zero.

## Recall accuracy per run

| Run | sliding_window_2 | recall_lab_brief_window_2 |
|---|---|---|
| run 1 | 0.00 | 1.00 |
| run 2 | 0.00 | 1.00 |
| run 3 | 0.00 | 0.80 |
| run 4 | 0.00 | 1.00 |
| run 5 | 0.00 | 0.60 |

## Spread

- sliding_window_2: min 0.00, max 0.00, mean 0.00
- recall_lab_brief_window_2: min 0.60, max 1.00, mean 0.88

## Per-question pass count across runs

### sliding_window_2

- 0/5  What color should you prioritize for me?
- 0/5  What food restriction should you remember for my daughter?
- 0/5  Where should you ship my order now?
- 0/5  Where did I use to ship orders before?
- 0/5  What kind of books do I like?

### recall_lab_brief_window_2

- 5/5  What color should you prioritize for me?
- 5/5  What food restriction should you remember for my daughter?
- 4/5  Where should you ship my order now?
- 3/5  Where did I use to ship orders before?
- 5/5  What kind of books do I like?

## Judge audit

The judge ran 3 calls per answer. 1 of 50 graded answers got a split verdict.

Split verdicts:
- run 1, sliding_window_2: "What kind of books do I like?" votes: drifted, honest_gap, honest_gap
