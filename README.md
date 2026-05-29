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
- budget-bounded sliding window for the equal-token-budget control
- full long context, planned where possible

The two-turn window makes the mechanism visible but is not a fair benchmark. The equal-token-budget control answers the obvious objection to it: a budget-bounded sliding window is given the same input-token budget Recall Lab actually spends, so the comparison stops being about prompt length. Both controls have now run on the relocation chain: the vector control plateaus around `0.55` and the budget-matched recency baseline sits near `0.04`, while Recall Lab holds `1.00`. See the status section for the per-question breakdown.

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
- ChromaDB for the flat vector-retrieval control

Deliberately small so the architecture, not the infra, is what is being measured.

All model calls go through one client factory (`recall_lab/llm.py`) with shared retries, a timeout, and OpenRouter provider routing. Provider routing is constrained for reproducibility: by default Azure is excluded, because its content filter once false-flagged a benign scenario prompt as a jailbreak and killed a run. Random provider routing otherwise adds variance unrelated to the memory strategy. The routing knobs live in `config.py` (`RECALL_OPENROUTER_IGNORE_PROVIDERS`, `RECALL_OPENROUTER_PROVIDER_ORDER`, `RECALL_OPENROUTER_ALLOW_FALLBACKS`).

## Status

Last update: May 26, 2026.

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
- Vector-retrieval control v10 on the relocation chain, four runs: sliding `0.00`, vector mean `0.55` (range `0.40`–`0.80`), Recall Lab `1.00`. The vector control passed the stable facts (color and shellfish) `4/4` but failed the chain: current city `1/4`, previous city `2/4`, first city `0/4`. Similarity retrieval keeps stable facts and cannot order superseded ones.
- Equal-token-budget control v10, five runs: Recall Lab `1.00`, budget-matched recency baseline mean `0.04`. Handing the recency baseline Recall Lab's full per-turn token budget did not close the gap.
- v10 judge audit: `0` split verdicts out of `110` graded answers across both campaigns.

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

# Add the flat vector-retrieval control (sliding + vector + recall)
python -m recall_lab.eval.multiday_trial --scenario scenarios/relocation_chain.json --agents all --verbose

# Run the equal-token-budget control: match the recency baseline to Recall Lab's budget
python -m recall_lab.eval.equal_budget_trial --scenario scenarios/relocation_chain.json --verbose
# Hand the recency baseline 1.5x Recall Lab's budget and see if it still loses
python -m recall_lab.eval.equal_budget_trial --scenario scenarios/relocation_chain.json --budget-scale 1.5 --verbose

# Run a 5-run variance campaign
python -m recall_lab.eval.variance --scenario scenarios/retail_memory_week.json --label v5_past_section
python -m recall_lab.eval.variance --scenario scenarios/relocation_chain.json --label v9_user_only_judge
```

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

Reports go under `reports/`. Report outputs are local artifacts and should not be committed unless needed for a public lab notebook.

## Current public read

The current honest claim is narrow:

> On two synthetic multi-day memory scenarios, Recall Lab's brief-backed memory with validity state and user-only salience held `1.00` recall while three baselines did not: a two-turn sliding window (`0.00`), flat vector retrieval (mean `0.55`, which keeps stable facts but cannot order a chain of corrections), and a sliding window given Recall Lab's full per-turn token budget (mean `0.04`). The result shows a mechanism, not a benchmark.

The vector control isolates the contribution of validity state: standard retrieval recalls un-superseded facts but fails the relocation chain. The equal-token-budget control shows the win is not prompt length: matching the recency baseline's budget to Recall Lab's does not close the gap.

Caveats that stay attached: one relocation scenario, four to five runs per control, no statistical test yet, and a single model family. This is not a general memory benchmark. The next steps are a long-context control, more scenarios, and the pre-registered 30-conversation protocol in `protocol.md`.

## Prior art note

Recall Lab does not claim to invent validity tracking, contradiction handling, or belief update. It adapts older ideas from truth maintenance systems, belief revision, temporal databases, and temporal knowledge graphs to the LLM agent memory setting, then tests whether those ideas reduce stale-memory failures in practice.
