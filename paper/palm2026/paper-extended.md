# Authority Is Not Recency: Validity-State Memory for Conversational Agents

**Target venue:** PALM Workshop @ NeurIPS 2026 (non-archival) — short paper track, 4 pages excl. references
**Deadline:** 24 August 2026, 11:59pm AoE
**Status:** DRAFT v1. Numbers are transcribed from `research-log.md`; every table is marked with the campaign version it came from. Verify against `reports/` before submission. Items marked `[CHECK]` need a number confirmed from the run artifacts.

---

## Abstract

Conversational memory systems decide what an agent believes about its user. Most production systems decide this with a proxy for authority: recency boosts, date-metadata filters, or "take the most recent value." We ask whether those proxies are sufficient, and show they are not — but only under conditions we can name precisely.

We build Recall Lab, a memory system that separates *retention* from *authority*: every user claim is consolidated into a trace carrying an explicit validity state (`active`, `superseded`), and superseded traces are rendered into a `Past, no longer current` section rather than deleted. Only user turns can grant a memory authority. We evaluate it against seven baselines spanning sliding windows, flat vector retrieval, a full industry retrieval stack (query rewriting, hybrid dense+BM25 with reciprocal rank fusion, recency boost, reranking), date-metadata filtering, a raw episodic read-time judge, and a deterministic `max(timestamp)` latest-value resolver.

We report three findings. **(1)** On a correction chain where every change is an explicit value-set, validity state is *unnecessary*: keep-everything, latest-wins, and validity state all tie at 1.00, while every retrieval baseline fails the oldest superseded fact. **(2)** The variable that determines whether retrieval recovers a correction chain is not retrieval quality, recency weight, context size, or visible timestamps — it is *presentation order*. Holding retrieval fixed and re-sorting snippets chronologically moves first-fact accuracy from 0/5 to 4/5. **(3)** On an adversarial scenario containing a value-less cancellation and a value-less re-mention, validity state is both necessary and sufficient: it is the only condition reaching 1.00, while the timestamp sort drops to 0.40 and the strongest retrieval stack to 0.72.

The unifying principle is that a statement should change memory only if it *sets a value*. Recency, timestamps, and similarity are proxies for authority that break precisely where intent and recency diverge. We also report a failure our own system introduced and how it was found and fixed, because consolidation can promote the wrong thing in a way that simpler baselines cannot.

---

## 1. Introduction

An assistant that remembers its user must answer a question retrieval does not pose: *of the things this user has told me, which are still true?*

The field largely answers this with recency. Vector stores add a recency boost. Production memory layers filter by date added. Agent frameworks resolve conflicts by taking the most recently stated value. These are all proxies for *authority* — the property of being the current, user-sanctioned truth — and they work well when every correction is an explicit restatement of a value.

We show they break in two specific places, both of which occur constantly in real conversation:

- **Value-less cancellation.** "Cancel that change, leave it as it was." The most recent statement about the attribute is the one being cancelled. A timestamp sort returns the cancelled value.
- **Value-less re-mention.** "Green is still such a beautiful color, I always come back to it." The user is reminiscing, not setting a preference. A timestamp sort — and a naive consolidation pipeline — lifts the stale value back to current.

Both cases share a structure: **the most recent mention of an attribute is not an assertion of its current value.** Recency cannot see this, because recency reads *when* a thing was said, not *whether the user was setting it*.

Our contributions:

1. **A memory architecture that makes authority explicit.** Validity state (`active` / `superseded`) is tracked per memory trace; superseded traces stay retrievable under an explicit `Past, no longer current` heading. Forgetting removes authority, not history.
2. **A controlled comparison against seven baselines**, including the two strongest non-retrieval alternatives that are usually omitted: keep-everything-and-read, and deterministic latest-value resolution.
3. **A negative result we pre-registered and report.** On our first scenario, validity state does *not* beat the simpler baselines. We say so, and identify the scenario property that makes it unnecessary.
4. **An isolation of presentation order** as the decisive variable for retrieval-based memory on temporal chains.
5. **An adversarial scenario** that separates a validity decision from a timestamp sort, on which only validity state holds.

---

## 2. Related work

**Memory rewriting.** Prior work reports that repeatedly rewriting a memory store degrades it, motivating keep-everything designs [arXiv:2605.12978]. We include a raw episodic read-time judge as a first-class baseline for exactly this reason, and find it competitive on accuracy but linearly growing in input-token cost.

**Retrieval-augmented memory.** Standard practice composes memory from top-k similarity search, often with hybrid lexical+dense retrieval, reciprocal rank fusion, recency weighting, and reranking. We implement this full stack rather than comparing against flat top-k, so that our claim is not made against a strawman.

**Date-metadata filtering.** Filtering retrieval by a recency window is a widely recommended pattern for personal memory. We test it directly and find it actively harmful (§5.1).

`[CHECK]` — Expand with 4–6 citations to current agent-memory systems before submission. This section is deliberately thin in the draft.

---

## 3. Method

Recall Lab maintains three memory surfaces: a raw **episodic log** (evidence), a machine-readable **trace store** (the substrate consolidation reasons over), and a **consolidated brief** (what the agent actually reads).

```
user turn
  → episodic log
  → user-only salience judge
  → memory trace store
  → contradiction check
  → validity state
  → activation ranking
  → consolidated brief
  → agent answer
```

A **sleep job** runs consolidation after each simulated day:

1. Read that day's episodic exchanges.
2. Score the **user turn only** for salience. The agent's own turn is never passed to the salience judge, so the system cannot grant authority to its own output.
3. Convert high-salience user claims into compressed memory traces.
4. Compare each new trace against active traces (contradiction check).
5. Confirm duplicates, correct stale traces, or add new traces.
6. Mark corrected memories `superseded` — never delete.
7. Render active traces as current memory.
8. Render superseded traces under `Past, no longer current` with explicit lineage labels.

**Validity states.** `active` — allowed to count as current truth. `superseded` — historical, retrievable, explicitly not current. The brief separates these into distinct sections so the agent can answer both "where do you ship now?" and "where did you ship before?" without conflating them.

**The value-setting guard.** Following the failure described in §5.3, salience scoring gates on whether the user turn *assigns the attribute as a current value*. A sentiment or reminiscence mention of an attribute is classified `UNRELATED`, not `CORRECTION`. A secondary guard at the contradiction classifier provides defence in depth. This is a general rule, not a scenario-specific patch.

### 3.1 Baselines

| Condition | Description |
|---|---|
| `sliding_window_2` | Last N turns only. |
| `budgeted_window` | Recency window sized to Recall Lab's per-turn input-token budget, so shorter prompts cannot win on length alone. |
| `vector_topk_5` | Flat top-k similarity over raw exchanges. No validity state. |
| `strong_rag` | Query rewriting → hybrid dense (ChromaDB) + BM25 → reciprocal rank fusion (k=60) → recency boost on turn index → LLM reranking. No validity state by design. |
| `strong_rag_dated` | As above, with real `added_at` timestamps, timestamp-based recency, and a metadata filter by date added (`recency_window_days=1.5`). |
| `episodic_judge` | Keep every statement verbatim, inject the whole log every turn, decide currency at read time. No consolidation. |
| `deterministic` | LLM extracts value-setting `(attribute, value)` pairs from user turns; current value per attribute chosen by `max(added_at, turn)` in pure Python. No model judgement of currency. |
| `recall_lab_brief` | The mechanism under test. |

`deterministic` is the sharpest objection to our thesis made concrete: *if the current answer is always the most recently stated value, why classify contradictions at all?*

### 3.2 Scenarios

Both scenarios are 4-day customer-assistant conversations with a fixed persona, ending in a 5-question recall evaluation.

**`relocation_chain`** — an ordered correction chain. Shipping city changes across three cities; two stable facts (favorite color, shellfish allergy) never change. Evaluation asks for the current, previous, and first city, plus both stable facts. Every change is an explicit, unambiguous value-set.

**`correction_intent`** — adversarial, same shape. Three discriminators:
- **Stale re-assertion** (expect *blue*): color changed green→blue, then green fondly re-mentioned in passing without changing the preference.
- **Revert** (expect *Berlin*): shipping Berlin→Munich, then the Munich change cancelled without restating Berlin.
- **Implicit correction by negation** (expect *father*): the attribute is corrected without being restated as `attribute=value`.

Plus two controls: a confirmation-that-is-not-a-change (shellfish, explicitly reaffirmed) and a history question (original color before the change).

### 3.3 Measurement

Recall accuracy is graded by an LLM judge with a **3-call audit**: three independent grading calls per answer, with split verdicts logged. Across the campaigns reported here, split verdicts were 0–1 per 175 graded answers.

**Reproducibility.** Model calls are provider-pinned, because routing a model across providers non-deterministically adds variance unrelated to the memory architecture. Agent model `openai/gpt-4o-mini` pinned to OpenAI; judge and contradiction classifier `anthropic/claude-sonnet-4.6` pinned to Anthropic. Fallbacks off — a pinned call the provider cannot serve errors rather than silently rerouting. Azure ignored on every call after a content-filter false positive killed a variance run. 5 seeds per condition per scenario.

---

## 4. Results

### 4.1 The correction chain: retrieval fails, and validity state is not needed

**Table 1.** `relocation_chain`, 5 seeds, pinned provider (campaign v12).

| Condition | Recall accuracy | Mean chat input tokens/turn |
|---|---|---|
| `sliding_window_2` | 0.00 | 267 |
| `budgeted_window` (equal budget) | 0.08 | matched |
| `vector_topk_5` | 0.52 | 373 |
| `strong_rag` | 0.76 | 479 |
| `episodic_judge` | **1.00** | 974 |
| `recall_lab_brief` | **1.00** | 438 |

`strong_rag` per-question: current city 4/5, previous city 5/5, **first city 0/5**, color 5/5, shellfish 5/5. A full industry retrieval stack recovers the *recent* end of a correction chain and collapses on the *oldest* superseded fact, because recency approximates authority near the present and decays into the past.

Adding the deterministic resolver (campaign v15, 5 seeds) produces the finding we pre-registered as a falsification target:

**Table 2.** `relocation_chain`, 5 seeds, with the latest-value resolver (v15).

| Condition | Recall accuracy | Spread |
|---|---|---|
| `sliding_window_2` | 0.00 | flat |
| `strong_rag_dated` | 0.40 | flat |
| `vector_topk_5` | 0.48 | 0.40–0.60 |
| `strong_rag` | 0.76 | 0.60–0.80 |
| `episodic_judge` | **1.00** | flat |
| `deterministic` | **1.00** | flat |
| `recall_lab_brief` | **1.00** | flat |

**Three very different strategies tie at the ceiling.** On this scenario, validity-state consolidation is not justified over the two simpler baselines. Every change here is a clean, explicit, user-stated value update with an unambiguous attribute — exactly the case where "take the latest" is correct by construction. We report this rather than bury it: the scenario was built to expose retrieval's failure to order superseded facts, and it does that well; it was not built to separate validity reasoning from a timestamp sort, and it does not.

### 4.2 Date-metadata filtering is actively harmful

**Table 3.** `relocation_chain`, n=4 (campaign v13).

| Condition | Recall accuracy | Mean tokens |
|---|---|---|
| `strong_rag` (turn recency) | 0.80 | 473 |
| `strong_rag_dated` (date filter) | **0.40** | 477 |

Per-question, `strong_rag_dated`: current city 4/4, previous city 0/4, first city 0/4, color 4/4, **shellfish allergy 0/4**.

The date filter bought a perfect current-city answer by deleting every fact older than the window — including a shellfish allergy stated on day 1 that *never changed*. A hard date filter cannot distinguish "old but still true" from "old and superseded," so it deletes both. This is the cleanest demonstration in our study that recency and date heuristics are proxies for authority rather than authority. **Authority requires knowing a fact was superseded, not merely that it is old.**

### 4.3 The decisive variable is presentation order, not retrieval quality

To remove the objection that §4.1's first-city failure is an artifact of two hyperparameters, we ablate recency weight and top-k and add a full-context "fair shot" with visible timestamps, in both relevance order and chronological order. Real BM25 hybrid retrieval throughout.

**Table 4.** Fair RAG sweep, `relocation_chain`, 5 seeds (campaign v14).

| Config | Mean acc | First-city | Mean tokens |
|---|---|---|---|
| `strong_rw0.0_k5` | 0.60 | 0/5 | 440 |
| `strong_rw0.3_k5` | 0.68 | 0/5 | 429 |
| `strong_rw0.6_k5` | 0.75 | 0/4 | 406 |
| `strong_rw0.3_k10` | 0.80 | 0/4 | 628 |
| `strong_rw0.3_kFULL` | 0.76 | 0/5 | 902 |
| `strong_fairshot` (relevance order) | 0.80 | 0/5 | 969 |
| `strong_fairshot_chrono` (time order) | **0.96** | **4/5** | 979 |
| — reference: `recall_lab_brief` | 1.00 | 5/5 | 438 |

Pre-registered predictions and outcomes:

1. First-city stays near 0 across every recency weight — **confirmed** (0/5, 0/5, 0/4). The failure is not a recency artifact.
2. First-city accuracy rises with top-k — **wrong**. `kFULL` stayed 0/5. The fact was present in context and still missed; it was never a coverage problem.
3. Full context with visible timestamps approaches 1.00 — **wrong** for relevance order (0.80, first-city 0/5). Visible dates were not enough.
4. Keeping the whole history orders the chain — **partial**. Necessary but not sufficient.

**Same retrieval, same timestamps, same context size: re-sorting oldest-first took first-city from 0/5 to 4/5 and overall accuracy from 0.80 to 0.96.** The bottleneck is that relevance ranking scrambles chronology and the model does not reliably re-sort by date on its own. Chronological presentation fixes it at keep-everything token cost (979 vs. the brief's 438). The validity brief reaches the same accuracy at bounded cost because it stores the ordered lineage explicitly rather than reconstructing it per query.

### 4.4 The adversarial scenario: validity state is necessary and sufficient

**Table 5.** `correction_intent`, 5 seeds, post-fix (campaign v16).

| Condition | Recall accuracy |
|---|---|
| `sliding_window_2` | 0.20 |
| `strong_rag_dated` | 0.36 |
| `deterministic` | 0.40 |
| `vector_topk_5` | 0.68 |
| `strong_rag` | 0.72 |
| `episodic_judge` | 0.92 |
| `recall_lab_brief` | **1.00** |

Per-discriminator:

**Revert** (value-less cancellation, expect *Berlin*): `recall_lab` 5/5. `deterministic` 0/5 — `max(timestamp)` returns Munich, as predicted by construction. Every retrieval baseline 0/5 — retrieval cannot represent a cancellation at all. `episodic_judge` 3/5 `[CHECK: confirm post-fix episodic revert count from reports/]` — reading the whole raw log does not force a single authority decision.

**Stale re-assertion** (value-less re-mention, expect *blue*): `recall_lab` 5/5 post-fix. `deterministic` 0/5 — the late green mention lifts the stale value.

The two cases that defeat a timestamp sort share one property: **the most recent mention sets no value.** A cancellation and a reminiscence are both statements *about* an attribute that do not assign it. Authority is about intent, not recency.

### 4.5 Cost

Across the v15+v16 campaigns: 5,797 model calls, $3.35 total. The **judge** accounts for 92% of spend ($3.08, 2,166 calls); the agent is $0.27 across 3,631 calls. Evaluation cost, not agent volume, dominates and scales with campaign size. Mean latency 1.43s, p95 2.97s.

---

## 5. Discussion

### 5.1 A two-scenario story

`relocation_chain` shows validity state is **not needed** when every change is an explicit value-set: keep-everything, latest-wins, and validity state all reach 1.00. `correction_intent` shows it is **needed and sufficient** when changes are adversarial: only validity state holds at 1.00, and the two failures it uniquely solves are both value-less statements.

Stating the negative result is what makes the positive one meaningful. A paper that only ran `correction_intent` would over-claim; a paper that only ran `relocation_chain` would conclude validity state is over-engineered. Both are true under different conditions, and the condition that separates them is nameable: **whether corrections in the conversation are explicit value-sets.**

### 5.2 Why retrieval cannot close this gap

Retrieval selects *candidates* by similarity. Authority decides which candidate is *current*. Every mechanism we tested that tries to derive authority from a retrieval-side signal — recency boost, date metadata, larger context, visible timestamps — either fails on the oldest superseded fact or, in the date-filter case, deletes stable facts along with stale ones. Only explicit chronological reconstruction recovers the chain, and it costs the entire history in context on every turn.

### 5.3 Consolidation can promote the wrong thing

Before the value-setting guard, `recall_lab` scored **0.64** on `correction_intent` — *below* `strong_rag` (0.80) and `episodic_judge` (0.92) — and 0/5 on the stale re-assertion it was designed to catch.

Trace analysis over the full Langfuse export located the failure precisely, and corrected our initial diagnosis. We first logged it as a contradiction-classifier bug. The traces showed the failure begins one layer earlier: the **salience judge** scored the reminiscence line at 0.55 with the reason *"User expresses a stable aesthetic preference for the color green"* and promoted it as a preference. The classifier then received "favorite color is green" against active "blue" and correctly returned `CORRECTION` — locally right, given a wrong input. Root cause: salience promoting a sentiment as a value-setting preference.

The guard moved the fix upstream. Post-fix: blue 0/5→5/5, Berlin 5/5→5/5 (the guard did not over-suppress real changes), green-as-history 1/5→5/5 (the lineage stopped inverting). The baselines were unchanged (deterministic 0.40, episodic 0.92), confirming the scenario did not get easier — `recall_lab` moved.

We report this because it is a real cost of consolidation: a system that decides what to remember can decide wrongly in a way that keep-everything cannot. The mitigation is a general rule about value-setting, and the failure was only findable because every model call was traced.

We also record a **false alarm** caught during the same analysis. An automated pass claimed the shellfish never-repeat rule was being silently demoted to the `Past` section — a safety bug. Direct inspection of all 300 briefs in the export: 207 had it correctly under `Things to never repeat`, 0 demoted. The apparent hits were the prompt's own instructions mentioning the section. Surprising automated claims were checked against full data before entering our log.

---

## 6. Limitations

We state these plainly; the study is small and the claims are scoped to it.

- **Two scenarios, one persona, four simulated days each.** The planned campaign (30 conversations, 200 evaluation questions per condition) has not been run. Results here are 5 seeds per condition per scenario.
- **One agent model** (`gpt-4o-mini`). We do not know how much of the retrieval failure is model-specific — particularly the finding that the model does not re-sort scrambled chronology on its own, which a stronger reasoner may do.
- **LLM-as-judge grading**, audited with 3 calls but not validated against human annotation.
- **No statistical testing.** The planned paired Wilcoxon signed-rank analysis and bootstrap CIs require the full campaign; spreads are reported instead.
- **`correction_intent` was authored by us** after observing that `relocation_chain` failed to discriminate. It is designed to expose a specific mechanism, and its discriminators are the two cases we predicted would separate validity state from a timestamp sort. Independent adversarial scenarios are needed.
- Several planned controls remain unbuilt: compressed vector retrieval, a full long-context oracle, and random-curated and human-curated brief controls that would isolate curation quality from brief format.

---

## 7. Conclusion

Memory systems for conversational agents are widely built on recency as a proxy for authority. We show the proxy holds exactly while corrections are explicit value-sets, and breaks on the two common cases where the most recent mention of an attribute assigns no value to it: cancellations and reminiscences. Making validity state explicit — and letting only user turns that *set a value* grant authority — resolves both, at bounded token cost, where a full industry retrieval stack, a date-metadata filter, keep-everything, and a deterministic latest-value resolver each fail at least one.

The principle we would carry forward: **forgetting should remove authority, not erase history** — and authority should be granted by intent, not by timestamp.

---

## Reproducibility

Code, scenarios, per-run reports, and a dated research log recording every pre-registration and its outcome (including the predictions we got wrong) are available at `[REPO URL]`. All model calls are provider-pinned; the pinning configuration is in `protocol.md`.

---

## Submission checklist

- [ ] Confirm `[CHECK]` numbers against `reports/`
- [ ] Fill §2 with 4–6 real citations; verify the arXiv id for the memory-rewriting paper
- [ ] Cut to 4 pages in the workshop template (this draft runs long — §4.2 and §4.5 compress first, §5.3 is worth keeping)
- [ ] Anonymize if the workshop uses double-blind review (check — PALM CFP)
- [ ] Public repo URL, or state "available on request"
- [ ] Confirm PALM submission portal + format (OpenReview?)
