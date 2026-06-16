# Recall Lab

A research repo testing whether selective consolidation, validity state, and source-aware memory help an agent stay coherent over long conversations.

Recall Lab is the experiment behind the Becoming Mind series. It is not a product or a benchmark yet. It is a small lab for finding memory failure modes, fixing them, and rerunning the same scenario until the mechanism is clear.

## Hypothesis

Coherence over long conversations may depend less on keeping every recent token and more on deciding:

- what should become memory
- what is still current truth
- what is historical but no longer current
- what should never be used as user truth

The working hypothesis is:

> Forgetting should remove authority, not erase history.

## Current architecture

Recall Lab has three memory surfaces:

- a raw episodic log for evidence
- a machine-readable trace store for consolidation
- a consolidated brief for the agent to read

```text
user turn
  -> episodic log
  -> user-only salience judge
  -> memory trace store
  -> contradiction check
  -> validity state
  -> activation ranking
  -> consolidated brief
  -> RecallAgent answer
```

The sleep job is the consolidation pass. In the current experiments it runs after each simulated day.

Every sleep pass:

1. Reads that day's episodic exchanges.
2. Scores the user turn for salience. The agent turn is not passed to the salience judge.
3. Converts high-salience user claims into compressed memory traces.
4. Compares each new trace against active traces.
5. Confirms duplicates, corrects stale traces, or adds new traces.
6. Marks corrected memories as superseded.
7. Renders active traces as current memory.
8. Renders superseded traces into `Past, no longer current` with explicit lineage labels.

The brief is what the agent reads. The trace store is what the sleep job reasons over.

Rendered diagrams of the pipeline, memory layers, validity state machine, source boundary, and v10 results are in [`diagrams/`](diagrams/README.md).

## Memory states

```text
active      allowed to count as current truth
superseded  no longer current, but still readable as history
archived    retired from the brief
```

This distinction matters. A stale memory should stop controlling behavior, but the agent may still need it for history questions.

## Memory layers

```text
Working memory      current user turn plus a small recent-turn buffer
Episodic log        every exchange, raw, in SQLite
Memory trace store  promoted semantic memories with salience and validity state
Consolidated brief  active and historical memories rendered as markdown
```

The brief currently has these sections:

- Stable facts about the user
- Active intents
- Open commitments
- Corrections
- Things to never repeat
- Past, no longer current

## Controls

Same conversational task, different memory strategy:

- naive sliding window
- Recall Lab brief-backed agent
- flat vector retrieval (ChromaDB, standard-RAG default)
- strong RAG: a production-grade pipeline with query rewriting, hybrid dense and BM25 retrieval fused with Reciprocal Rank Fusion, a recency boost, and a reranker
- strong RAG with date metadata (`strong_rag_dated`): timestamp recency plus a metadata-by-date filter
- budget-bounded sliding window for the equal-token-budget control
- raw episodic read-time judge (keep everything, decide at read time)
- deterministic latest-value resolver (extract pairs, pick winner by max timestamp)
- full long context, planned where possible

The two-turn window makes the mechanism visible but is not a fair benchmark. The strong RAG control answers the strongest objection, that the failure is just a weak retriever. It is the stack a serious team ships, and on the relocation chain it still cannot reconstruct a correction chain. The equal-token-budget control answers the prompt-length objection: a budget-bounded sliding window is given the same input-token budget Recall Lab actually spends.

On the relocation chain under a pinned provider, the headline numbers are sliding window `0.00`, flat vector `0.52`, strong RAG `0.76`, raw episodic `1.00`, and Recall Lab's validity brief `1.00`. Strong RAG recovered the recent links of the chain and missed the first city every run. A fair sweep (`eval/fair_rag_sweep.py`) then varied the recency weight and top_k and handed the model the whole conversation with timestamps visible: the first city still failed under relevance ordering, and only reordering the same context chronologically recovered it (`4/5`). The fix was order of presentation, not retrieval quality. See the status section and `reports/` for the per-question breakdowns.

## Metrics

- recall accuracy on follow-up questions
- failure mode: correct, hallucinated, drifted, or honest gap
- judge disagreement audit during variance runs
- estimated output tokens
- promoted memories
- corrected memories
- active vs superseded traces

The output token count is currently an approximation from generated text length, not billing data.

## Stack

- Python 3.11+
- OpenAI client pointed at OpenRouter
- SQLite for the episodic log
- JSONL for memory traces
- Markdown for the consolidated brief
- pytest for pure-function checks
- ChromaDB for the dense half of retrieval
- rank-bm25 for the lexical half of the strong RAG hybrid retrieval

Deliberately small so the architecture, not the infra, is what is being measured.

All model calls go through one client factory (`recall_lab/llm.py`) with shared retries, a timeout, and OpenRouter provider routing. Provider routing is constrained for reproducibility: by default Azure is excluded, because its content filter once false-flagged a benign scenario prompt as a jailbreak and killed a run. Random provider routing otherwise adds variance unrelated to the memory strategy. The routing knobs live in `config.py` (`RECALL_OPENROUTER_IGNORE_PROVIDERS`, `RECALL_OPENROUTER_PROVIDER_ORDER`, `RECALL_OPENROUTER_ALLOW_FALLBACKS`).

## Status

Last update: June 9, 2026.

Working now:

- `EpisodicLog` persists raw exchanges to SQLite, fetches one UTC day of exchanges, and records promoted rows with salience scores.
- `SlidingWindowAgent` runs as the recent-turn baseline.
- `RecallAgent` reads the consolidated brief before each answer, keeps a bounded recent-turn buffer, calls OpenRouter, and appends every response to the episodic log.
- `consolidation/activation.py` scores retrievability with an ACT-R style activation function.
- `consolidation/interference_check.py` reproduces the stale-memory failure where re-reference makes an old corrected fact stronger.
- `consolidation/contradiction.py` classifies memories as CONFIRM, CORRECT, or UNRELATED and supports reversible supersession.
- `memory/traces.py` stores promoted memories with salience, status, references, supersession links, and brief sections.
- `memory/traces.py` renders superseded memories into `Past, no longer current` with explicit labels such as `Earliest past` and `Most recent past before current`.
- `memory/brief.py` loads, renders, deduplicates, and saves the consolidated memory brief.
- `consolidation/judge.py` scores only the user turn. The agent turn is hidden from the salience judge so the system cannot promote its own generated text as user truth.
- `consolidation/sleep.py` routes promoted memories through salience, trace creation, contradiction checks, validity transitions, activation ranking, and brief rendering.
- `controls/vector.py` implements the flat vector-retrieval control: each exchange is embedded in an isolated in-memory ChromaDB collection and the top-k most similar are retrieved per turn. It has no validity state, so a superseded fact and its correction compete on similarity alone. The embedding function is injectable for offline, key-free tests.
- `controls/budgeted.py` implements a sliding window bounded by an input-token budget instead of a turn count, for the equal-token-budget comparison.
- `controls/episodic.py` implements the raw episodic read-time-judge control: keep every statement verbatim, inject the whole log each turn, and ask the model to work out the current answer. No consolidation runs. This is the "just keep everything" baseline from arxiv 2605.12978, which found that repeatedly rewriting memory degrades it. It tests whether validity-state consolidation beats raw retention, on accuracy and on the growing input-token cost.
- `controls/deterministic.py` implements the deterministic latest-value resolver: an LLM extracts value-setting (attribute, value) pairs from each user turn, and the current value per attribute is chosen by Python `max(timestamp)`, never a model judgement. It answers the sharpest objection to validity state, "why not just take the latest?", and isolates where a timestamp sort breaks down versus a real validity decision (a confirmation misread as a change, an implicit correction). The extractor is injectable for offline, key-free tests.
- `controls/strong_rag.py` implements the industry-standard RAG control: query rewriting, hybrid dense plus BM25 retrieval fused with Reciprocal Rank Fusion, a recency boost, and a reranker. A `strong_rag_dated` variant adds real date metadata, timestamp recency, and a metadata-by-date filter, and a `show_timestamps` and `chronological` option control whether the retrieved snippets carry dates and what order they are presented in. Every external piece is injectable, so the plumbing is tested offline without a key.
- `eval/fair_rag_sweep.py` runs the fair RAG sweep: a recency-weight ablation, a top_k sweep, and a fair-shot config that hands the model the whole conversation with timestamps visible, in relevance order and in chronological order, to isolate whether the chain failure is retrieval reach or order of presentation.
- `eval/equal_budget_trial.py` measures Recall Lab's mean input tokens per turn, then runs the budget-matched recency baseline on the same scenario and reports accuracy at equal token budget.
- `multiday_trial.py` now records an input-token estimate per turn and supports the vector control via `--agents vector` or `--agents all`.
- `eval/metrics.py` uses an LLM scorer with a stricter rubric. Variance runs can audit each answer with three judge calls.
- `eval/multiday_trial.py` runs configurable multi-day scenario files and writes JSON plus markdown reports.
- `eval/variance.py` runs the same scenario several times and reports spread, per-question pass counts, and judge disagreement.
- `scenarios/retail_memory_week.json` defines the simple retail correction trial.
- `scenarios/relocation_chain.json` defines the harder Lagos to Berlin to Nairobi chain.
- `tests/` has pure-function coverage for activation, brief rendering, contradiction state, scoring helpers, and sleep trace integration.

Observed so far:

- Sliding window forgot a turn-1 favorite color by turn 5.
- Activation alone failed under re-reference pressure: an old Lagos fact beat a newer Berlin correction.
- Stronger contradiction classification made current truth more stable.
- Rendering superseded traces into `Past, no longer current` fixed the simple history gap.
- The salience judge originally saw both user and agent turns. That let the agent's own answer become new memory. The user-only judge fixed that source leak.
- Retail trial v5: sliding window `0.00`, Recall Lab `1.00` across five runs.
- Relocation-chain trial v9: sliding window `0.00`, Recall Lab `1.00` across five runs.
- In v9, Recall Lab passed current city, previous city, first city, color, and safety restriction at `5/5` each.
- v9 judge audit: `0` split verdicts out of `50` graded answers.
- Vector-retrieval control, pinned-provider rerun (v11) on the relocation chain, five runs: sliding `0.00`, vector `0.40` (range `0.40`–`0.40`), Recall Lab `1.00`. The vector control passed the stable facts (color and shellfish) `5/5` but failed the chain: current city `0/5`, previous city `0/5`, first city `0/5`. Similarity retrieval keeps stable facts and cannot order superseded ones.
- Equal-token-budget control, pinned-provider rerun (v11), five runs: Recall Lab `1.00`, budget-matched recency baseline `0.00`. Handing the recency baseline Recall Lab's full per-turn token budget did not close the gap.
- Provider pin tightened the spread. Pre-pin (v10) the vector control ranged `0.40`–`0.80` (mean `0.55`) and the recency baseline `0.00`–`0.20` (mean `0.04`); pinned (v11) both are flat at `0.40` and `0.00`. Part of the pre-pin variance was provider-routing noise, not the memory strategy. Recall Lab held `1.00` in both.
- Judge audit across the pinned reruns: `1` split verdict out of `125` graded answers.
- Chapter 3 lineup (v12) on the relocation chain, five runs: sliding `0.00`, flat vector `0.52`, strong RAG `0.76`, raw episodic `1.00`, Recall Lab `1.00`. Strong RAG recovered the current and previous city and missed the first city `0/5`.
- Fair RAG sweep (v14): the first city stayed at or near zero across recency weights `0.0`–`0.6` and across top_k `5`, `10`, and full context, including full context with timestamps visible in relevance order. Reordering the same context chronologically recovered it (`4/5`). The chain failure was order of presentation, not retrieval reach.
- Date-metadata filtering (`strong_rag_dated`, v13) did not help: it perfected the current city and dropped older facts, including a stable allergy, because a date filter cannot tell an old-but-true fact from a superseded one.

Still incomplete:

- full long-context control
- larger eval set beyond the current two scenarios
- brief decay policy
- human-curated and random-curated brief controls
- UI or screenshot dashboard beyond markdown reports
- statistical protocol over many conversations

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

# Set your OpenRouter key
echo 'OPENROUTER_API_KEY=sk-or-...' > .env

# Run tests
python -m pytest

# Run the activation interference check
python -m recall_lab.consolidation.interference_check

# Run contradiction validity checks
python -m recall_lab.consolidation.contradiction

# Run the first v0.1 head-to-head eval
python -m recall_lab.eval.v01_head_to_head

# Run the default configurable trial
python -m recall_lab.eval.multiday_trial

# Run both agents on a named scenario
python -m recall_lab.eval.multiday_trial --scenario scenarios/retail_memory_week.json --agents both --verbose
python -m recall_lab.eval.multiday_trial --scenario scenarios/relocation_chain.json --agents both --verbose

# Add every control in one run: sliding, vector, strong RAG, strong RAG with date
# metadata, raw episodic, and the Recall Lab brief
python -m recall_lab.eval.multiday_trial --scenario scenarios/relocation_chain.json --agents all --verbose

# Run the equal-token-budget control: match the recency baseline to Recall Lab's budget
python -m recall_lab.eval.equal_budget_trial --scenario scenarios/relocation_chain.json --verbose
# Hand the recency baseline 1.5x Recall Lab's budget and see if it still loses
python -m recall_lab.eval.equal_budget_trial --scenario scenarios/relocation_chain.json --budget-scale 1.5 --verbose

# Run a 5-run variance campaign
python -m recall_lab.eval.variance --scenario scenarios/retail_memory_week.json --label v5_past_section
python -m recall_lab.eval.variance --scenario scenarios/relocation_chain.json --agents all --label v12_chapter3_lineup
```

### Chapter 3 experiments

These reproduce the strong-RAG and order findings from Becoming Mind Chapter 3. The key is read from `.env`, so no export is needed. Run from the repo root, in a real terminal (background jobs can be killed mid-campaign), and tune `RUNS` down to smoke-test first.

```bash
# Full lineup plus the equal-budget control, five seeds, pinned provider
RUNS=5 bash scripts/run_chapter3.sh

# The fair RAG sweep: recency-weight ablation, top_k sweep, and the chronological
# fair-shot that shows order of presentation is the lever. Token-heavy, so it
# defaults to a 1-call judge and RUNS=3.
RUNS=5 bash scripts/run_fair_rag.sh
```

Both write a `*_summary.md` you can read straight from `reports/`. The committed summaries in this repo were produced by these scripts.

## Scenario editing

Scenarios live in `scenarios/*.json`.

You can change an experiment without editing core code:

- add or remove days
- change the number of exchanges
- edit user messages
- change final evaluation questions
- change expected answers
- change the working-window size

The runner reads the scenario file and keeps the same experiment logic.

## Outputs

`recall_lab.eval.multiday_trial` writes:

- `trial_result.json`: exact machine-readable result
- `trial_report.md`: screenshot-friendly report

`recall_lab.eval.variance` writes:

- `variance_summary.md`: per-run accuracy, spread, per-question pass count, and judge audit
- `run_*/trial_result.json`
- `run_*/trial_report.md`

`recall_lab.eval.fair_rag_sweep` writes:

- `fair_rag_summary.md`: per-config accuracy, mean input-token cost, and per-question pass count

Reports go under `reports/`, which is gitignored, so raw per-run outputs stay local. The campaign summaries (`variance_summary.md` and `fair_rag_summary.md`) are force-added and committed, so the numbers behind the claims are visible in the repo without carrying every run.

## Current public read

The current honest claim is narrow:

> On two synthetic multi-day memory scenarios, with model calls pinned to one provider per family for reproducibility, Recall Lab's brief-backed memory held `1.00` recall while the baselines did not: a two-turn sliding window (`0.00`), flat vector retrieval (`0.52`), and even a production-grade RAG stack with query rewriting, hybrid retrieval, reranking, and recency (`0.76`). The strong RAG stack recovered the recent links of a correction chain and missed the first one every run. A fair sweep showed the cause was order of presentation, not retrieval quality: only reordering the same retrieved context chronologically recovered the first fact, and only by carrying the whole history at a token cost the bounded brief avoids. The result shows a mechanism, not a benchmark.

The strong RAG control isolates the point: standard and even well-engineered retrieval recall un-superseded facts but cannot order a chain of corrections, because similarity ranking has no notion of sequence. The equal-token-budget control shows the win is not prompt length: matching the recency baseline's budget to Recall Lab's does not close the gap.

Caveats that stay attached: one relocation scenario, four to five runs per control, no statistical test yet, and a single model family. This is not a general memory benchmark. The next steps are a long-context control, more scenarios, and the pre-registered 30-conversation protocol in `protocol.md`.

## Prior art note

Recall Lab does not claim to invent validity tracking, contradiction handling, or belief update. It adapts older ideas from truth maintenance systems, belief revision, temporal databases, and temporal knowledge graphs to the LLM agent memory setting, then tests whether those ideas reduce stale-memory failures in practice.
