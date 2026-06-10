# Fair RAG sweep, relocation_chain

5 seeds per config. Same scenario and pipeline; only the named knobs change.

Reference (v12 5-seed, not re-run): flat vector 0.52, raw episodic 1.00 at ~974 tok, validity brief 1.00 at ~438 tok.

## Accuracy and cost

| Config | mean acc | mean chat input tokens | runs |
|---|---|---|---|
| strong_rw0.0_k5 | 0.60 | 440 | 5 |
| strong_rw0.3_k5 | 0.68 | 429 | 5 |
| strong_rw0.6_k5 | 0.75 | 406 | 4 |
| strong_rw0.3_k10 | 0.80 | 628 | 4 |
| strong_rw0.3_kFULL | 0.76 | 902 | 5 |
| strong_fairshot | 0.80 | 969 | 5 |
| strong_fairshot_chrono | 0.96 | 979 | 5 |

## Per-question pass count

### strong_rw0.0_k5
- 2/5  Where should you ship my order now?
- 3/5  Which city did I ship to right before my current one?
- 0/5  What was the very first city I shipped to?
- 5/5  What is my favorite color?
- 5/5  What food restriction should you remember for my daughter?

### strong_rw0.3_k5
- 4/5  Where should you ship my order now?
- 3/5  Which city did I ship to right before my current one?
- 0/5  What was the very first city I shipped to?
- 5/5  What is my favorite color?
- 5/5  What food restriction should you remember for my daughter?

### strong_rw0.6_k5
- 3/4  Where should you ship my order now?
- 4/4  Which city did I ship to right before my current one?
- 0/4  What was the very first city I shipped to?
- 4/4  What is my favorite color?
- 4/4  What food restriction should you remember for my daughter?

### strong_rw0.3_k10
- 4/4  Where should you ship my order now?
- 4/4  Which city did I ship to right before my current one?
- 0/4  What was the very first city I shipped to?
- 4/4  What is my favorite color?
- 4/4  What food restriction should you remember for my daughter?

### strong_rw0.3_kFULL
- 4/5  Where should you ship my order now?
- 5/5  Which city did I ship to right before my current one?
- 0/5  What was the very first city I shipped to?
- 5/5  What is my favorite color?
- 5/5  What food restriction should you remember for my daughter?

### strong_fairshot
- 5/5  Where should you ship my order now?
- 5/5  Which city did I ship to right before my current one?
- 0/5  What was the very first city I shipped to?
- 5/5  What is my favorite color?
- 5/5  What food restriction should you remember for my daughter?

### strong_fairshot_chrono
- 5/5  Where should you ship my order now?
- 5/5  Which city did I ship to right before my current one?
- 4/5  What was the very first city I shipped to?
- 5/5  What is my favorite color?
- 5/5  What food restriction should you remember for my daughter?
