# Recall Lab

A research repo testing whether selective forgetting improves agent coherence over long conversations.

## Hypothesis

Coherence over long conversations may depend more on what an agent removes, compresses, and marks stale than on how much context it can hold.

Recall Lab tests that idea with small controlled conversations first, then larger multi-day trials.

## Current architecture

Recall Lab now has two memory surfaces:

- a raw episodic log for evidence
- a consolidated brief for the agent to read

Between those two surfaces sits a machine-readable trace store. That is where salience, contradiction, activation, and validity state are handled.

```text
user exchanges
  -> episodic log
  -> salience judge
  -> memory trace store
  -> contradiction check
  -> validity state
  -> activation ranking
  -> consolidated brief
  -> RecallAgent answer
```

The sleep job is the consolidation pass. In the current experiments it runs after a simulated day, not literally at real bedtime.

Every sleep pass:

1. Reads that day's episodic exchanges.
2. Scores unpromoted exchanges for salience.
3. Converts high-salience exchanges into compressed memory traces.
4. Compares each new trace against the strongest active traces.
5. Confirms, corrects, or adds the trace.
6. Marks corrected memories as superseded.
7. Renders only active traces into the brief.

The brief is what the agent reads. The trace store is what the sleep job reasons over.

## Memory layers

```text
Working memory      current user turn plus a small recent-turn buffer
Episodic log        every exchange, raw, in SQLite
Memory trace store  promoted semantic memories with salience and validity state
Consolidated brief  active memories rendered as markdown for the agent
```

## Controls

Same conversational task, different memory strategy:

- naive sliding window
- Recall Lab brief-backed agent
- flat vector retrieval, planned
- full long context, planned where possible

## Metrics

- recall accuracy on follow-up questions
- failure mode: correct, hallucinated, drifted, or honest gap
- estimated output tokens
- promoted memories
- corrected memories
- active vs superseded traces

The token count is currently an approximation from output text length, not billing data.

## Stack

- Python 3.11+
- OpenAI client pointed at OpenRouter
- SQLite for the episodic log
- JSONL for memory traces
- Markdown for the consolidated brief
- ChromaDB planned for the vector control

Deliberately small so the architecture, not the infra, is what is being measured.

## Prior art note

Recall Lab does not claim to invent validity tracking, contradiction handling, or belief update. It adapts older ideas from truth maintenance systems, belief revision, temporal databases, and temporal knowledge graphs to the LLM agent memory setting, then tests whether those ideas reduce stale-memory failures in practice.

## Status

Last update: May 24, 2026.

Working now:

- `EpisodicLog` persists raw exchanges to SQLite, fetches one UTC day of exchanges, and records promoted rows with salience scores.
- `SlidingWindowAgent` runs end to end through OpenRouter.
- `consolidation/activation.py` scores retrievability with an ACT-R style activation function.
- `consolidation/interference_check.py` reproduces the stale-memory failure where re-reference makes an old corrected fact stronger.
- `consolidation/contradiction.py` classifies memories as CONFIRM, CORRECT, or UNRELATED and supports reversible supersession.
- `memory/traces.py` stores promoted memories as machine-readable traces with salience, status, references, supersession links, and brief sections.
- `memory/brief.py` loads, renders, deduplicates, and saves the consolidated memory brief.
- `consolidation/judge.py` scores exchanges through OpenRouter and returns normalized salience verdicts.
- `consolidation/sleep.py` now routes promoted memories through salience, trace creation, contradiction checks, validity transitions, activation ranking, and brief rendering.
- `RecallAgent` reads the consolidated brief before each answer, keeps a bounded recent-turn buffer, calls OpenRouter, and appends every response to the episodic log.
- `recall_demo.py` compares sliding window against the brief-backed Recall agent on the favorite-color failure.
- `eval/v01_head_to_head.py` turns that comparison into a repeatable one-scenario eval.
- `eval/multiday_trial.py` runs configurable multi-day scenario files and writes JSON plus markdown reports.
- `scenarios/retail_memory_week.json` defines the current four-day retail shopping assistant trial.

Observed so far:

- Sliding window forgot a turn-1 favorite color by turn 5.
- Recall Lab remembered the favorite color after the sleep job promoted it into the brief.
- Activation alone failed under re-reference pressure: an old Lagos fact beat a newer Berlin correction.
- Validity state now exists in the main sleep path, so corrected traces can be superseded before the brief is rendered.
- A Recall-only Day 1 retail staged run completed: 6 exchanges, 4 promoted, 4 active traces, 2 skipped, final eval accuracy 0.6 after only one simulated day.

Still incomplete:

- full four-day retail trial results
- sliding window vs Recall Lab comparison on the full retail trial
- vector retrieval control
- full long-context control
- UI or screenshot dashboard beyond the markdown report
- larger eval set beyond the current retail scenario
- brief decay policy

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Set your OpenRouter key
echo 'OPENROUTER_API_KEY=sk-or-...' > .env

# Run the first sliding-window baseline demo
python demo.py

# Run the activation interference check
python -m recall_lab.consolidation.interference_check

# Run the contradiction validity checks
python -m recall_lab.consolidation.contradiction

# Compare sliding window against the brief-backed Recall agent
python recall_demo.py

# Run the first v0.1 head-to-head eval
python -m recall_lab.eval.v01_head_to_head

# Run the configurable retail memory-week trial
python -m recall_lab.eval.multiday_trial

# Safer staged run while developing
python -m recall_lab.eval.multiday_trial --agents recall --max-days 1 --verbose
python -m recall_lab.eval.multiday_trial --agents recall --max-days 2 --verbose
python -m recall_lab.eval.multiday_trial --agents recall --verbose
python -m recall_lab.eval.multiday_trial --agents both --verbose
```

## Scenario editing

The retail trial is configured in `scenarios/retail_memory_week.json`.

You can change the experiment without editing core code:

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

By default these go under `reports/retail_memory_week/`. Report outputs are local artifacts and should not be committed unless needed for a public lab notebook.

## Prior art note

Recall Lab does not claim to invent validity tracking, contradiction handling, or belief update. It adapts older ideas from truth maintenance systems, belief revision, temporal databases, and temporal knowledge graphs to the LLM agent memory setting, then tests whether those ideas reduce stale-memory failures in practice.
