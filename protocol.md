# Recall Lab Protocol

Status: draft protocol, to be completed before first results are published.

## Research question

Does validity-state memory consolidation with user-only salience improve long-horizon agent coherence and honest failure compared with sliding-window, budget-matched recency, and flat vector-retrieval baselines?

## Hypothesis

An agent that consolidates high-salience user turns into a brief, tracks each memory's validity state (active, superseded, archived), and renders superseded memories into a `Past, no longer current` section will maintain coherence over long conversations with fewer hallucinated and stale-memory recall failures than agents that rely on recency windows or flat vector retrieval. The mechanism under test is validity state plus user-only salience: only user turns can grant a memory authority, and a corrected fact loses authority without being erased.

## Conditions

Built today:

- Recall Lab brief-backed memory: user-only salience consolidation with validity state and a `Past, no longer current` section. The mechanism under test.
- Sliding window: last N turns only.
- Budget-matched sliding window: a recency window sized to Recall Lab's per-turn input-token budget, so the comparison is not about prompt length.
- Flat vector retrieval: top-k similar raw exchanges, no validity state.

Planned, not built yet (one-line cost to build):

- Compressed vector retrieval: top-k extracted semantic facts. Cost: an extraction pass over the log plus an embed-and-store step on the compressed facts, roughly a day.
- Full long-context oracle: full conversation in context where it fits. Cost: a thin agent that concatenates the whole log, plus per-model context-limit handling, half a day.
- Random-curated brief: a brief filled with randomly selected exchanges at Recall Lab's size. Cost: a sampler over the log, a few hours. Isolates curation quality from brief format.
- Human-curated brief: a brief hand-written by a person from the same log. Cost: a labeling tool plus annotator time, the largest of the planned items.

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

## Future work

Not part of the current hypothesis or campaign. Each is its own job.

- Decay. A retrievability decay over the brief, so quiet stale memories fade without a correction. `DECAY_HALF_LIFE_DAYS` exists in config and an ACT-R style activation function exists, but the brief decay policy raises `NotImplementedError`. When built, the test is whether brief-with-decay beats brief-without-decay on honest-failure, the original decay hypothesis.
- Learned authority. Today authority is rule-based: a user correction supersedes the old fact. Whether authority can be learned from data instead is open.
- Multi-agent memory. Shared or per-agent validity state across more than one agent.
- The planned conditions above that are not built: compressed vector retrieval, the full long-context oracle, and the random-curated and human-curated brief controls.
