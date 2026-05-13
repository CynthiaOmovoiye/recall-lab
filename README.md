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

Scaffold, May 12, 2026. Stubs in place. First runs targeted for week of May 18. First lab notebook post on LinkedIn targeted May 28.

## See also

- Research notes and hypothesis tree: `/Users/cynthiaomovoiye/Documents/Claude/Projects/Profile Visibility/posts/03_memory_engineering_research_NOTES.md`
- Public series: Becoming Mind on LinkedIn (chapter 1 drops May 13, 2026)

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Set your OpenRouter key
echo 'OPENROUTER_API_KEY=sk-or-...' > .env

# (once stubs are filled in)
python -m recall_lab.eval.harness --agent recall --conversation sample
```

Most commands will not work until the stubs are filled in. The architecture is what matters at this stage.
