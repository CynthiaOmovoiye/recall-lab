# Recall Lab Protocol

Status: active. Two scenarios have been run to completion under the pinned-provider setup (v12–v16); the 30-conversation campaign in Planned dataset has not. Last synced August 19, 2026.

## Research question

Does validity-state memory consolidation with user-only salience improve long-horizon agent coherence and honest failure compared with sliding-window, budget-matched recency, and flat vector-retrieval baselines?

## Hypothesis

An agent that consolidates high-salience user turns into a brief, tracks each memory's validity state (active, superseded, archived), and renders superseded memories into a `Past, no longer current` section will maintain coherence over long conversations with fewer hallucinated and stale-memory recall failures than agents that rely on recency windows or flat vector retrieval. The mechanism under test is validity state plus user-only salience: only user turns can grant a memory authority, and a corrected fact loses authority without being erased.

### Amendment, June 17, 2026: the value-setting guard

The v16 adversarial campaign found that user-only salience is necessary but not sufficient. A user turn that mentions an attribute without assigning it — a fond re-mention of a superseded value — was promoted by the salience judge as a current preference, and the contradiction classifier then correctly labelled it `CORRECT` against the active trace, inverting the lineage. The failure originated at the salience layer, not the classifier.

The mechanism under test is therefore amended to: **a statement grants or changes authority only if it sets a value.** Salience distinguishes a value-setting statement from sentiment or reminiscence about a value; a re-mention that does not assign the attribute as current is `UNRELATED`. A reminiscence guard at the contradiction classifier is the secondary defence, not the fix. This is a general rule about intent, not a scenario-specific patch, and it is the form of the hypothesis that the `correction_intent` results test.

### Runtime invariant checking

`memory/invariants.py` defines the brief invariants (principally: no trace may be simultaneously active and past). Since commit `a3492f7`, `run_sleep_job` runs `check_invariants` after every consolidation pass. Non-strict by default: violations are reported in the summary dict under `invariant_violations` and warned to stdout, but do not raise, so one bad pass cannot abort a campaign and destroy the other runs' data. `strict=True` raises `BriefInvariantError`, for tests and CI.

## Conditions

Built today:

- Recall Lab brief-backed memory: user-only salience consolidation with validity state and a `Past, no longer current` section. The mechanism under test.
- Sliding window: last N turns only.
- Budget-matched sliding window: a recency window sized to Recall Lab's per-turn input-token budget, so the comparison is not about prompt length.
- Flat vector retrieval: top-k similar raw exchanges, no validity state. The naive RAG baseline.
- Strong RAG: the industry-standard retrieval stack. Query rewriting, hybrid dense plus BM25 retrieval fused with Reciprocal Rank Fusion, a recency boost on the fused scores, and a reranking pass before the top-k context is composed. Still no validity state by design. It isolates whether the best engineered retrieval can order superseded facts, or whether recency heuristics approximate authority without replacing it. This is the control that turns "RAG misses authority" into a measured claim rather than a claim against a strawman. Code: `recall_lab/controls/strong_rag.py`. Two configurations run side by side: `strong_rag` uses turn-order recency; `strong_rag_dated` uses real date metadata, timestamp recency plus metadata filtering by date added (`recency_window_days`, default 1.5 days from config), so the claim covers explicit date filtering and not only a recency boost.
- Raw episodic read-time judge: keep every statement verbatim, inject the whole log each turn, decide the current answer at read time. No consolidation. The "just keep everything" baseline from arxiv 2605.12978, which reported that repeatedly rewriting memory degrades it. Tests whether consolidation beats raw retention on accuracy and on input-token cost as the log grows.
- Deterministic latest-value resolver: an LLM extracts value-setting (attribute, value) pairs from each user turn; the current value per attribute is chosen by `max(timestamp)` in Python, with no model judgement of currency. The "why not just take the latest?" baseline. If it matches Recall Lab on the relocation chain, validity state is over-engineered for this scenario; where it breaks (a confirmation misread as a change, an implicit correction, a mis-split attribute) isolates what a validity decision buys over a timestamp sort.
- Fair RAG sweep (`eval/fair_rag_sweep.py`): not a single condition but a methodology over the strong RAG control. It ablates the recency weight (`0.0`, `0.3`, `0.6`), sweeps top_k (`5`, `10`, full context), and runs a fair-shot config that gives the model the entire conversation with timestamps printed on each record, in relevance order and in chronological order. The purpose is to remove the bias question: it shows the first-city failure is not an artifact of the two hyperparameters, and isolates order of presentation as the variable that actually moves it. The v14 result: the first city fails across every recency weight and top_k, including full context with timestamps in relevance order, and only chronological ordering recovers it (`4/5`). See the June 9 entries in `research-log.md`.

Planned, not built yet (one-line cost to build):

- Compressed vector retrieval: top-k extracted semantic facts. Cost: an extraction pass over the log plus an embed-and-store step on the compressed facts, roughly a day.
- Full long-context oracle: full conversation in context where it fits. Cost: a thin agent that concatenates the whole log, plus per-model context-limit handling, half a day.
- Random-curated brief: a brief filled with randomly selected exchanges at Recall Lab's size. Cost: a sampler over the log, a few hours. Isolates curation quality from brief format.
- Human-curated brief: a brief hand-written by a person from the same log. Cost: a labeling tool plus annotator time, the largest of the planned items.

## Scenarios run

Both are four-day customer-assistant conversations with a fixed persona, ending in a five-question recall evaluation. Files in `scenarios/`.

**`relocation_chain`** — an ordered correction chain. The shipping city changes across three cities; two facts (favourite colour, a shellfish restriction) are stated once and never change. Evaluation asks for the current, previous and first city, plus both stable facts. Every change is an explicit, unambiguous, user-stated value-set.

Purpose: expose whether retrieval can order a chain of superseded facts. It does that well. It does **not** separate validity-state reasoning from a timestamp sort, because "take the latest" is correct by construction when every change is an explicit value-set. See the falsification record below.

**`correction_intent`** — adversarial, same shape, built after v15 to discriminate what `relocation_chain` could not. Three discriminators:

- **Stale re-assertion** (expect *blue*): colour changed green → blue, then green fondly re-mentioned in passing without changing the preference. The late mention lifts a superseded value under any recency or timestamp rule.
- **Revert** (expect *Berlin*): shipping Berlin → Munich, then the Munich change cancelled without restating Berlin. The most recent value-set is Munich, so a timestamp sort returns the cancelled value.
- **Implicit correction by negation** (expect *father*): the attribute is corrected without being restated as `attribute=value`.

Plus two controls: a confirmation-that-is-not-a-change (the food restriction, explicitly reaffirmed) and a history question (the original colour before the change).

Both discriminators share one property, and it is the property the amended hypothesis names: **the most recent mention of the attribute sets no value.**

Note on interpretation: `sliding_window_2` scores 5/5 on the current-colour question here purely because a two-turn window happens to span the relevant turn. That is a coverage artifact and carries no information about recency windows.

## Falsification record

The falsification target below was written before results. One falsification event has occurred and is recorded here rather than only in the log.

**v15, June 16, 2026 — falsified for `relocation_chain`.** The deterministic latest-value resolver reached 1.00, matching Recall Lab and the raw episodic judge. Three very different strategies tied at the ceiling, so that scenario does not justify validity-state consolidation over the two simpler non-retrieval baselines. Reported, not buried. The response was not to weaken the claim but to build the scenario that discriminates (`correction_intent`), and to state the condition under which validity state is unnecessary: when every change is an explicit value-set.

**v16 post-fix, June 17, 2026 — held for `correction_intent`.** Recall Lab is the only condition at 1.00 (deterministic 0.40, strong RAG 0.72, raw episodic 0.92). It uniquely handles both the revert and the stale re-assertion.

The scoped claim is therefore the **pair**, not either scenario alone: validity state is unnecessary when corrections are explicit value-sets, and necessary and sufficient when they are not. Any writeup that reports only `correction_intent` over-claims; any writeup that reports only `relocation_chain` concludes the mechanism is over-engineered. Both are required.

## Planned dataset

- 30 conversations
- 200 evaluation questions per condition
- 5 random seeds per conversation

The final dataset generation method will be written here before results are published.

## Planned metrics

- Recall accuracy
- Tokens per response
- Failure mode
- Honest-failure rate

Failure labels:

- correct
- abstains correctly
- admits uncertainty
- falsely claims memory
- fabricates
- retrieves stale memory
- overgeneralizes from the brief

## Planned controls

All agent variants will run on the same conversations and evaluation questions.

An equal-token-budget run will be included so shorter prompts do not win only because they use fewer tokens.

## Planned analysis

- Paired Wilcoxon signed-rank test at the conversation level
- 1,000-resample conversation-level bootstrap for confidence intervals
- Multiple-comparison correction to be specified before results

## Pre-registration notes

Before results are published, this file should include:

- model names and versions
- provider names
- temperature and decoding settings
- retrieval top-k
- embedding model
- salience threshold
- validity-transition and supersession policy
- prompt templates
- scoring scripts
- dataset generation method

### Provider routing

Model calls are pinned for reproducibility, because OpenRouter routes a model across providers non-deterministically and that adds variance unrelated to the memory architecture.

- Pinned provider for the agent model (`openai/gpt-4o-mini`): `OpenAI`. OpenAI is one of only two providers that serve this model on OpenRouter, and it runs no extra content filter on top of OpenAI's own moderation.
- Pinned provider for the judge and contradiction classifier (`anthropic/claude-sonnet-4.6`): `Anthropic`. The agent pin cannot serve an Anthropic model, so each model family pins to its own provider. Both land on one provider every run, which is what makes the campaign reproducible: a reviewer can say which provider scored which run.
- Fallbacks: off. A pinned call that the provider cannot serve errors instead of silently rerouting. Transient blips are still absorbed by client-level retries.
- Ignored providers: `Azure`, on every call. The May 29 research-log entry records an Azure content-filter false-positive that flagged a benign shopping-assistant prompt as a jailbreak and killed a variance run.

These values live in `recall_lab/config.py` and `.env.example`: `RECALL_OPENROUTER_PROVIDER_ORDER=OpenAI`, `RECALL_OPENROUTER_JUDGE_PROVIDER_ORDER=Anthropic`, `RECALL_OPENROUTER_ALLOW_FALLBACKS=false`, `RECALL_OPENROUTER_IGNORE_PROVIDERS=Azure`.

## Falsification target

If the validity-state brief shows no improvement over flat vector retrieval and the budget-matched recency baseline on honest-failure metrics across the 30-conversation campaign, the validity-state hypothesis is falsified for this setup.

Added after v15, and the stricter test: retrieval baselines are no longer the bar that matters. The hypothesis is falsified if the brief shows no improvement over the **deterministic latest-value resolver and the raw episodic judge** on a scenario whose corrections are not all explicit value-sets. Beating retrieval alone is not sufficient evidence for validity state, because v15 showed a timestamp sort can match the brief wherever the retrieval baselines fail. See Falsification record.

## Future work

Not part of the current hypothesis or campaign. Each is its own job.

- Decay. A retrievability decay over the brief, so quiet stale memories fade without a correction. `DECAY_HALF_LIFE_DAYS` exists in config and an ACT-R style activation function exists, but the brief decay policy raises `NotImplementedError`. When built, the test is whether brief-with-decay beats brief-without-decay on honest-failure, the original decay hypothesis.
- Learned authority. Today authority is rule-based: a user correction supersedes the old fact. Whether authority can be learned from data instead is open.
- Multi-agent memory. Shared or per-agent validity state across more than one agent.
- The planned conditions above that are not built: compressed vector retrieval, the full long-context oracle, and the random-curated and human-curated brief controls.
