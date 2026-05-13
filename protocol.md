# Recall Lab Protocol

Status: draft protocol, to be completed before first results are published.

## Research question

Does selective consolidation with decay improve long-horizon agent coherence and honest failure compared with common memory baselines?

## Hypothesis

An agent that promotes high-salience exchanges into a consolidated brief and lets low-salience exchanges stop entering the prompt will maintain coherence over long conversations with fewer hallucinated recall failures than agents that only use recency windows or raw vector retrieval.

## Conditions

- Recall Lab: LLM-judged consolidation with decay
- Sliding window: last N turns only
- Raw vector retrieval: top-k similar raw exchanges
- Compressed vector retrieval: top-k extracted semantic facts
- Full long context: full conversation when it fits
- Consolidated brief without decay
- Random-curated brief
- Human-curated brief

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
- decay policy
- prompt templates
- scoring scripts
- dataset generation method

## Falsification target

If the consolidated-brief-with-decay variant shows no improvement over the consolidated-brief-without-decay variant on honest-failure metrics, the selective-consolidation-with-decay hypothesis is falsified for this setup.
