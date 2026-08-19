# Authority Is Not Recency: Validity-State Memory for Conversational Agents

*Target: PALM Workshop @ NeurIPS 2026 · short paper, 4 pages excl. references · non-archival · deadline 24 Aug 2026*
*All figures verified against `reports/variance/*/variance_summary.md` on 19 Aug 2026. Extended version with full pre-registrations: `paper-extended.md`.*

---

## Abstract

Conversational memory systems decide what an agent believes about its user, and most decide it with a proxy for authority: a recency boost, a date filter, or "take the most recent value." We ask when those proxies suffice.

Recall Lab separates retention from authority: user claims become traces carrying explicit validity state (`active`, `superseded`), superseded traces are rendered under `Past, no longer current` rather than deleted, and only user turns that *set a value* may grant authority. We compare against seven baselines, including a full retrieval stack (query rewriting, hybrid dense+BM25 with reciprocal rank fusion, recency boost, reranking), date-metadata filtering, a raw episodic read-time judge, and a deterministic `max(timestamp)` resolver.

Three findings. **(1)** Where every correction is an explicit value-set, validity state is *unnecessary*: keep-everything, latest-wins and validity state all tie at 1.00, while every retrieval baseline fails the oldest superseded fact. **(2)** What decides whether retrieval recovers a correction chain is not retrieval quality, recency weight, context size or visible timestamps, but *presentation order*: re-sorting the same snippets chronologically moves first-fact accuracy from 0/5 to 4/5. **(3)** On an adversarial scenario with a value-less cancellation and a value-less re-mention, only validity state reaches 1.00; the timestamp sort falls to 0.40 and the strongest retrieval stack to 0.72.

The principle: a statement should change memory only if it sets a value. We also report a failure our own consolidation introduced, and how tracing located it.

---

## 1. Introduction

An assistant that remembers its user must answer a question retrieval does not pose: *of the things this user has told me, which are still true?*

The field largely answers with recency. Vector stores add recency boosts; memory layers filter by date added; agent frameworks resolve conflicts by taking the latest value. These are proxies for **authority** — being the current, user-sanctioned truth — and they work while every correction explicitly restates a value.

They break in two places, both common in real conversation:

- **Value-less cancellation.** *"Cancel that change, leave it as it was."* The most recent statement about the attribute is the one being cancelled; a timestamp sort returns the cancelled value.
- **Value-less re-mention.** *"Green is still such a beautiful color, I always come back to it."* The user is reminiscing, not setting a preference; a timestamp sort lifts a superseded value back to current.

Both share a structure: **the most recent mention of an attribute is not an assertion of its current value.** Recency reads *when* something was said, not *whether the user was setting it*.

Contributions: (i) a memory architecture making authority explicit, where forgetting removes authority rather than history; (ii) a controlled comparison against seven baselines, including the two strongest non-retrieval alternatives usually omitted — keep-everything and deterministic latest-value; (iii) **a pre-registered negative result we report rather than bury**, naming the scenario property under which validity state is unnecessary; (iv) isolation of presentation order as the decisive variable for retrieval-based memory; (v) an adversarial scenario on which only validity state holds.

## 2. Related work

**Consolidation can degrade memory.** Zhang et al. [1] show that LLM-driven consolidation degrades over time — utility rises, then falls below a no-memory baseline — and that an episodic-only control retaining raw trajectories stays competitive with every consolidator they test. We therefore make `episodic_judge` a first-class baseline rather than a strawman, and §5 reports that our own consolidator independently reproduced their finding before we fixed it: pre-guard, the brief scored 0.64 against the episodic control's 0.92. Our claim is not that consolidation is safe, but that a nameable rule about *what may be consolidated* separates the two regimes.

**Memory systems and benchmarks.** Production memory layers extract, consolidate and retrieve salient facts from ongoing conversation into a persistent store [2]; Du [3] surveys the mechanism families and the shift in evaluation from static recall toward multi-session agentic tests, naming continual consolidation and learned forgetting as open problems. Superseded facts are already a recognized failure axis: LongMemEval [4] carries a *knowledge updates* task category alongside temporal reasoning and abstention, and LoCoMo [5] evaluates very long-term multi-session dialogue. Those benchmarks establish that the problem is real and measure how often systems fail it; we hold retrieval fixed and vary only how authority is decided, to isolate *which* mechanism is responsible.

**Position and order effects.** Liu et al. [6] show that models use information in the middle of a long context less reliably than at its edges. Our §4.2 result is adjacent but distinct: enlarging the context did not recover the missed fact, and visible timestamps did not either. What recovered it was re-sorting the *same* retrieved snippets chronologically — an ordering effect on temporal reasoning, over and above positional degradation.

**Prior art.** Recall Lab does not claim to invent validity tracking. It adapts truth maintenance [7] and belief revision [8] to the LLM agent memory setting and tests whether they reduce stale-memory failures in practice.

## 3. Method

Recall Lab maintains a raw **episodic log** (evidence), a **trace store** (what consolidation reasons over), and a **consolidated brief** (what the agent reads). A **sleep job** runs after each simulated day: score the *user turn only* for salience — the agent's own turn is never scored, so the system cannot grant authority to its own output; compress high-salience claims into traces; check each against active traces; confirm, correct, or add; mark corrected traces `superseded` rather than deleting; render active traces as current memory and superseded traces under `Past, no longer current` with explicit lineage labels. The brief separates these sections so the agent can answer both "where do you ship now?" and "where did you ship before?" without conflating them.

**The value-setting guard.** Following the failure in §5, salience gates on whether the user turn *assigns the attribute as a current value*; a sentiment or reminiscence mention is `UNRELATED`, not `CORRECTION`. A secondary guard sits at the contradiction classifier. This is a general rule about intent, not a scenario-specific patch. A brief-invariant check runs after every consolidation pass (principally: no trace may be simultaneously active and past), reporting violations into the pass summary.

**Baselines.** `sliding_window_2` (last N turns); `budgeted_window` (window sized to Recall Lab's token budget, so shorter prompts cannot win on length alone); `vector_topk_5` (flat top-k similarity); `strong_rag` (query rewriting → hybrid dense + BM25 → reciprocal rank fusion [9], k=60 → recency boost → LLM reranking; no validity state by design); `strong_rag_dated` (adds `added_at` timestamps, timestamp recency, and a metadata filter, `recency_window_days`=1.5); `episodic_judge` (inject the whole log every turn, decide currency at read time); `deterministic` (an LLM extracts value-setting `(attribute, value)` pairs; the current value is `max(added_at, turn)` in pure Python, no model judgement of currency). The last makes the sharpest objection to our thesis concrete: *if the current answer is always the most recently stated value, why classify contradictions at all?*

**Scenarios.** Both are 4-day customer-assistant conversations with a fixed persona, ending in a 5-question recall evaluation. **`relocation_chain`**: shipping city changes across three cities; two facts stated once and never changed. Every change is an explicit, unambiguous value-set. **`correction_intent`**: adversarial, built after the §4.1 result, with a *stale re-assertion* (color green→blue, then green fondly re-mentioned without changing the preference; expect blue), a *revert* (Berlin→Munich, then the Munich change cancelled without restating Berlin; expect Berlin), and an *implicit correction by negation*, plus two controls (a confirmation-that-is-not-a-change, and a history question).

**Measurement.** Recall accuracy graded by an LLM judge with a 3-call audit; 0 of 175 graded answers split in the campaign of §4.3. Model calls are provider-pinned, because routing a model across providers non-deterministically adds variance unrelated to the memory architecture: agent `gpt-4o-mini` pinned to OpenAI, judge and classifier `claude-sonnet-4.6` pinned to Anthropic, fallbacks off. 5 seeds per condition per scenario.

## 4. Results

### 4.1 Where corrections are explicit, validity state is unnecessary

**Table 1.** `relocation_chain`, 5 seeds, pinned provider.

| Condition | Recall acc. | Spread |
|---|---|---|
| `sliding_window_2` | 0.00 | flat |
| `budgeted_window` | 0.08 | flat |
| `strong_rag_dated` | 0.40 | flat |
| `vector_topk_5` | 0.48 | 0.40–0.60 |
| `strong_rag` | 0.76 | 0.60–0.80 |
| `episodic_judge` | **1.00** | flat |
| `deterministic` | **1.00** | flat |
| `recall_lab_brief` | **1.00** | flat |

`strong_rag` per-question: current city 4/5, previous 5/5, **first city 0/5**, both stable facts 5/5. A full industry stack recovers the *recent* end of a correction chain and collapses on the *oldest* superseded fact, because recency approximates authority near the present and decays into the past.

But **three very different strategies tie at the ceiling.** On this scenario validity state is not justified over the simpler baselines: every change is a clean, explicit, user-stated value update with an unambiguous attribute — exactly where "take the latest" is correct by construction. We pre-registered this as a falsification target and report it. The scenario exposes retrieval's failure to order superseded facts, which it does well; it was not built to separate validity reasoning from a timestamp sort, and it does not.

Cost separates them where accuracy does not: the bounded brief reaches 1.00 at 438 mean input tokens/turn against `episodic_judge`'s 974, and that gap widens with conversation length.

**Date filtering is actively harmful.** `strong_rag_dated` scores 0.40 against `strong_rag`'s 0.76, buying a perfect current-city answer (4/4) by deleting every fact older than the window — including a stable food restriction stated on day 1 that *never changed* (0/4). A hard filter cannot distinguish "old but still true" from "old and superseded," so it deletes both.

### 4.2 The decisive variable is presentation order

To remove the objection that the first-city failure is an artifact of two hyperparameters, we ablate recency weight and top-k, then give the model the entire conversation with timestamps visible, in relevance order and in chronological order. Real BM25 hybrid retrieval throughout.

**Table 2.** Fair RAG sweep, `relocation_chain`, 5 seeds.

| Config | Mean acc. | First-city | Tokens |
|---|---|---|---|
| `strong_rw0.3_k5` | 0.68 | 0/5 | 429 |
| `strong_rw0.6_k5` | 0.75 | 0/4 | 406 |
| `strong_rw0.3_kFULL` | 0.76 | 0/5 | 902 |
| `fairshot` (relevance order) | 0.80 | 0/5 | 969 |
| `fairshot_chrono` (time order) | **0.96** | **4/5** | 979 |
| *ref:* `recall_lab_brief` | 1.00 | 5/5 | **438** |

Three pre-registered predictions failed. First-city accuracy did *not* rise with top-k — `kFULL` stayed 0/5, so the fact was present in context and still missed; it was never a coverage problem. Full context with visible timestamps did *not* approach 1.00 in relevance order (0.80, 0/5); visible dates were not enough.

**Same retrieval, same timestamps, same context size: re-sorting oldest-first took first-city from 0/5 to 4/5.** The bottleneck is that relevance ranking scrambles chronology and the model does not reliably re-sort by date on its own. Chronological presentation fixes it at keep-everything token cost. The validity brief reaches the same accuracy at 45% of the tokens because it stores ordered lineage explicitly rather than reconstructing it per query.

### 4.3 Where corrections set no value, validity state is necessary and sufficient

**Table 3.** `correction_intent`, 5 seeds, post-fix.

| Condition | Recall acc. | Revert (Berlin) | Re-assertion (blue) |
|---|---|---|---|
| `sliding_window_2` | 0.20 | 0/5 | 5/5\* |
| `strong_rag_dated` | 0.36 | 0/5 | 3/5 |
| `deterministic` | 0.40 | 0/5 | 0/5 |
| `vector_topk_5` | 0.68 | 0/5 | 3/5 |
| `strong_rag` | 0.72 | 0/5 | 4/5 |
| `episodic_judge` | 0.92 | 3/5 | 5/5 |
| `recall_lab_brief` | **1.00** | **5/5** | **5/5** |

\* A coverage artifact: a two-turn window happens to span the relevant turn for this one question. It is not evidence about recency windows.

The timestamp sort that tied at 1.00 in Table 1 collapses to 0.40. Only the validity brief holds. On the revert, every retrieval baseline scores 0/5 — retrieval cannot represent a cancellation at all — and `deterministic` returns Munich exactly as predicted by construction. Reading the whole raw log does not force a single authority decision either: `episodic_judge` manages 3/5.

The two cases that defeat a timestamp sort share one property: **the most recent mention sets no value.** A cancellation and a reminiscence are both statements *about* an attribute that do not assign it.

## 5. Consolidation can promote the wrong thing

Before the value-setting guard, `recall_lab_brief` scored **0.64** on `correction_intent` — *below* `strong_rag` (0.80) and `episodic_judge` (0.92) — and 0/5 on the stale re-assertion it was designed to catch.

Trace analysis over the full export (5,797 model calls) located the failure and corrected our initial diagnosis. We first logged it as a contradiction-classifier bug. The traces showed it begins one layer earlier: the **salience judge** scored the reminiscence line at 0.55, reasoning *"User expresses a stable aesthetic preference for the color green,"* and promoted it as a preference. The classifier then received "favorite color is green" against active "blue" and returned `CORRECTION` — locally correct, given a wrong input. Root cause: salience promoting sentiment as a value-set.

The fix moved upstream. Post-fix: blue 0/5→5/5; Berlin 5/5→5/5, so the guard did not over-suppress real changes; the history question 1/5→5/5, recovered as a consequence once the lineage stopped inverting. Baselines were unchanged (deterministic 0.40, episodic 0.92), confirming the scenario did not get easier — Recall Lab moved.

We report this because it is a real cost of consolidation, and the one Zhang et al. [1] warn of: a system that decides what to remember can decide wrongly in a way keep-everything cannot. The mitigation is a general rule about value-setting, and the failure was only findable because every model call was traced. The same analysis also produced a false alarm we caught by checking a claimed safety regression against all 300 briefs in the export, where it did not appear.

## 6. Limitations

Two synthetic scenarios, one persona, four simulated days each, 5 seeds per condition; the pre-registered 30-conversation campaign has not been run. One agent model, so we cannot separate model-specific effects — particularly the §4.2 finding that the model does not re-sort scrambled chronology, which a stronger reasoner may do. LLM-as-judge grading, audited with 3 calls but not validated against human annotation. No statistical testing; spreads are reported instead of the planned Wilcoxon and bootstrap CIs. Critically, **`correction_intent` was authored by us** after observing that `relocation_chain` failed to discriminate, and its discriminators are the two cases we predicted would separate validity state from a timestamp sort; independently authored adversarial scenarios are needed. Several controls remain unbuilt: compressed vector retrieval, a long-context oracle, and random- and human-curated brief controls that would isolate curation quality from brief format.

## 7. Conclusion

Memory systems for conversational agents are widely built on recency as a proxy for authority. The proxy holds exactly while corrections are explicit value-sets, and breaks on the two common cases where the most recent mention of an attribute assigns no value to it: cancellations and reminiscences. Making validity state explicit — and letting only value-setting user turns grant authority — resolves both at bounded token cost, where a full retrieval stack, a date filter, keep-everything, and a deterministic latest-value resolver each fail at least one.

Forgetting should remove authority, not erase history; and authority should be granted by intent, not by timestamp.

**Reproducibility.** Code, scenarios, per-run reports, and a dated research log recording every pre-registration and its outcome — including the predictions we got wrong — at `[REPO URL]`.

---

## References

[1] D. Zhang, Y. Lin, Z. Wu, Y. Sun, B. Li, D. Li, and H. Peng. Useful Memories Become Faulty When Continuously Updated by LLMs. arXiv:2605.12978, May 2026.

[2] P. Chhikara, D. Khant, S. Aryan, T. Singh, and D. Yadav. Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory. arXiv:2504.19413, April 2025.

[3] P. Du. Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers. arXiv:2603.07670, March 2026.

[4] D. Wu, H. Wang, W. Yu, Y. Zhang, K.-W. Chang, and D. Yu. LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory. arXiv:2410.10813, October 2024.

[5] A. Maharana, D.-H. Lee, S. Tulyakov, M. Bansal, F. Barbieri, and Y. Fang. Evaluating Very Long-Term Conversational Memory of LLM Agents. arXiv:2402.17753, February 2024.

[6] N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni, and P. Liang. Lost in the Middle: How Language Models Use Long Contexts. *Transactions of the ACL*, 12:157–173, 2024. arXiv:2307.03172.

[7] J. Doyle. A Truth Maintenance System. *Artificial Intelligence*, 12(3):231–272, 1979.

[8] C. E. Alchourrón, P. Gärdenfors, and D. Makinson. On the Logic of Theory Change: Partial Meet Contraction and Revision Functions. *Journal of Symbolic Logic*, 50(2):510–530, 1985.

[9] G. V. Cormack, C. L. A. Clarke, and S. Buettcher. Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. In *SIGIR*, 2009.

---

## Pre-submission checklist

- [x] §2 citations — 9 references added, all verified against arXiv/publisher primary sources on 19 Aug 2026. `2605.12978` confirmed correct (Zhang et al., *Useful Memories Become Faulty When Continuously Updated by LLMs*).
- [ ] Confirm PALM submission portal and template (OpenReview?), and whether review is double-blind — anonymize `[REPO URL]` if so
- [ ] Typeset and confirm ≤4 pages excl. references. Body is ~2,450 words + 3 tables, which should land close. If it runs over, cut in this order: (1) the date-filter paragraph in §4.1, (2) Table 2 down to 4 rows, (3) the cost sentence closing §4.1.
- [ ] Decide whether to name agent/judge model versions inline or defer to a repro appendix
