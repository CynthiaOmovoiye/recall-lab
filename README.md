# Recall Lab

A research repo testing whether selective forgetting improves agent coherence over hundreds of turns.

## Hypothesis

Coherence over long conversations is a function of how much an agent throws away on purpose, not how much context it can hold or how cleverly it retrieves. The next leap in agent memory is architectural, not bigger windows.

## Architecture

Three memory layers, plus a nightly consolidation job.

```
Working memory    ──► current turn context
Episodic log      ──► every exchange, raw, on disk (never re-injected after the day)
Consolidated brief ──► living markdown the agent reads first every turn
```

Every "night" (eval interval), a sleep job runs:

1. Reads the day's episodic log
2. An LLM judge scores each exchange on salience
3. High-salience exchanges get rewritten as compressed semantic statements and merged into the brief
4. Low-salience exchanges stay on disk but stop being injected
5. Old brief entries decay if not reinforced (Forgetting Curves Lab, secondary)

## Controls

Same conversational task, three baselines plus the experimental agent:

- Naive sliding window
- Flat vector retrieval (Mem0-style)
- Full long context (when fits)
- Recall Lab (the experimental agent)

## Metrics

- Recall accuracy on follow-up questions about earlier turns
- Tokens per response
- Failure mode shape: drift, hallucinate, or correctly say "I do not remember"
- Coherence score on multi-day conversational tasks (LOCOMO subset where possible)

## Stack

- Python 3.11+
- OpenAI client pointed at OpenRouter (set `OPENROUTER_API_KEY`)
- ChromaDB for the vector control
- SQLite for the episodic log
- A single markdown file for the brief

Deliberately small so the architecture, not the infra, is what is being measured.

## Status

Week 1 build, May 17, 2026. Last update May 24, 2026.

Working now:

- `EpisodicLog` persists raw exchanges to SQLite, fetches one UTC day of exchanges, and records promoted rows with salience scores.
- `SlidingWindowAgent` runs end to end through OpenRouter.
- `demo.py` runs a five-turn synthetic conversation with a two-turn window and stores each exchange in a separate demo database.
- `consolidation/activation.py` scores how retrievable a memory trace is, using an ACT-R style decay function over creation recency, re-reference frequency, and base salience.
- `consolidation/interference_check.py` pits an old fact against a newer contradiction and prints which one wins retrieval.
- `consolidation/contradiction.py` is the validity half of memory. It classifies a new statement against an old fact as CONFIRM, CORRECT, or UNRELATED, moves a corrected fact from active to superseded, chains a supersedes pointer to its replacement, filters retrieval to current truth, and logs every transition so a suppression can be undone.
- `memory/brief.py` now loads, renders, deduplicates, and saves the consolidated memory brief.
- `consolidation/judge.py` now scores exchanges through OpenRouter and returns a normalized salience verdict.
- `consolidation/sleep.py` now runs a first end-to-end consolidation pass: fetch exchanges, skip already promoted rows, score new rows, promote high-salience statements into the brief, and mark promoted rows in SQLite.
- `RecallAgent` now reads the consolidated brief before each answer, keeps only a small recent-turn buffer, calls OpenRouter, and appends every response to the episodic log.
- `recall_demo.py` compares the two-turn sliding-window baseline against the brief-backed Recall agent on the favorite-color failure.
- `eval/v01_head_to_head.py` turns that comparison into a tiny repeatable eval with an expected answer, recall accuracy, failure mode, and token estimate.
- `eval/multiday_trial.py` runs configurable multi-day scenario files, so trial length, messages, final questions, and output paths can change without editing core logic.
- `memory/traces.py` stores promoted memories as machine-readable traces with salience, validity status, references, supersession links, and brief sections.
- The sleep job now routes promoted memories through the trace store, checks contradictions against the strongest active traces, updates validity state, applies activation ranking, and renders only active memories into the brief.

First observed baseline failure: a fact introduced on turn 1 was unavailable by turn 5 once it fell outside the two-turn window, even though the exchange still existed on disk.

Second observed failure, contradiction: an old fact ("User lives in Lagos", 40 days old) lost retrieval to a newer correction ("User moved to Berlin", 5 days old) on a clean slate. After the old fact was re-referenced three times, it won again. Decay handles quiet stale facts, but frequency can keep an outdated fact stronger than a newer correction. Full numbers are in the May 20 research log entry.

Still stubbed:

- Vector retrieval control
- Evaluation metrics
- Brief decay policy
- Contradiction-aware sleep job integration



## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Set your OpenRouter key
echo 'OPENROUTER_API_KEY=sk-or-...' > .env

# Run the first Week 1 baseline demo
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
```

`demo.py` exercises the sliding-window baseline and episodic log. `recall_demo.py` exercises the first working Recall Lab path: log, sleep job, brief, and brief-backed response.
