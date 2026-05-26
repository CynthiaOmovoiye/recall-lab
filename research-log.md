# Recall Lab research log

Created May 12, 2026.

Hypothesis: coherence over hundreds of turns is a function of selective forgetting more than context size or retrieval quality.

This log is appended to by the Tuesday and Saturday Recall Lab nudge routine and by Cynthia directly. Each entry should have a date, what was tried, what worked, what failed, and what to try next.

---

## May 12, 2026. Scaffold created.

Repo skeleton in place. Three layers stubbed: working memory, episodic log, consolidated brief. Sleep job stub uses an LLM judge against a salience threshold. Two control agents stubbed: sliding window and flat vector retrieval. Eval harness stub captures per-turn results.

Suggested next step: implement EpisodicLog.append and fetch_day against SQLite. Get a conversation flowing through SlidingWindowAgent end-to-end before anything else, so the eval harness has something to compare against.

---

## May 17, 2026. First Week 1 slice runs end to end.

Implemented `EpisodicLog.append`, `fetch_day`, and `mark_promoted` against SQLite. Verified the full round trip manually: write an exchange, fetch it back, mark it promoted with salience `0.9`, then fetch the updated row.

Implemented `SlidingWindowAgent.respond` with OpenRouter-backed chat calls. Ran a five-turn synthetic conversation with `window=2` using `openai/gpt-4o-mini`.

Observed first useful baseline failure: turn 1 stored "My favorite color is blue." By turn 5, the agent could not answer the favorite-color question because the fact had fallen outside the two-turn model context, even though the exchange still existed on disk.

Added `demo.py` to run the baseline conversation and persist each exchange to a separate demo SQLite database. A clean demo run stored five exchanges.

What worked:
- Raw episodic persistence
- Day-bounded exchange retrieval
- Promotion bookkeeping in the raw log
- End-to-end sliding-window baseline
- Reproducible first synthetic run

What is still stubbed:
- Salience judge
- Consolidated brief integration
- Recall agent response path
- Vector retrieval control
- Evaluation scoring

Suggested next step: implement the salience judge and score one synthetic conversation end to end before wiring the brief into the Recall agent.

---

## May 20, 2026. Activation and interference check.

Implemented an ACT-R style activation function for memory traces. Activation combines creation recency, re-reference frequency, base salience, and emotional weight into one retrievability score.

Added `recall_lab.consolidation.interference_check` to test an old fact against a newer contradiction:
- Old memory: "User lives in Lagos"
- New memory: "User moved to Berlin in April 2026"

Scenario 1, clean slate:
- Old fact: 40 days old, 0 re-references, activation `-2.633`
- New fact: 5 days old, 0 re-references, activation `-1.594`
- Winner: new

Scenario 2, old fact re-referenced:
- Old fact: 40 days old, re-referenced 3 times, activation `-0.063`
- New fact: 5 days old, 0 re-references, activation `-1.594`
- Winner: old

Read: decay handles quiet stale facts, but frequency can keep an outdated fact stronger than a newer correction. Activation can estimate memory strength. It cannot decide whether a re-reference confirms a memory or contradicts it.

Next design implication: contradiction handling belongs in the sleep job before reinforcement. If a user correction contradicts an older trace, the old trace should be marked superseded instead of reinforced.

---
## May 24, 2026. Contradiction and validity state.

Implemented `recall_lab.consolidation.contradiction` as the validity half of memory. This builds on the May 20 result: activation can rank memories, but it cannot tell whether a strong memory is still true.

What changed:
- Added `status` and `supersedes` fields to `MemoryTrace`.
- Added `MemoryStatus`: active, superseded, archived.
- Added `classify(old, new)` to label a new statement against an old fact as CONFIRM, CORRECT, or UNRELATED.
- Added `supersede()` to move an old trace from active to superseded and point the new trace back to it.
- Added `revert()` so a wrong suppression can be undone one transition at a time.
- Added `append_transition()` to write validity changes to `data/transitions.jsonl`.
- Added `current_traces()` so retrieval can filter for current truth before ranking with activation.

Offline checks passed: correction chains, one-step revert, CONFIRM guard, and transition logging. Live `classify()` checks passed for three pairs: Lagos to Berlin as CORRECT, Lagos to Lagos as CONFIRM, and Lagos to tab indentation as UNRELATED.

Read: rank and validity are separate. Activation decides which memory is strongest. Validity decides whether that memory is allowed to count as current truth. Retrieval should filter by validity first, then rank the survivors.

Next design implication: wire contradiction handling into the sleep job. On a CORRECT verdict, the old memory should become superseded instead of reinforced.

---

## May 24, 2026. Salience judge and brief consolidation slice.

Implemented the first working consolidation path after the episodic log. The brief can now load markdown sections, render them back to disk, add entries under canonical sections, and deduplicate exact repeats.

Implemented the salience judge through OpenRouter. Given one exchange, it returns a normalized `SalienceVerdict` with a score, reason, suggested brief section, and compressed statement. Low-salience exchanges below the threshold do not produce brief entries.

Ran a smoke test with two synthetic exchanges:
- High-salience: `My daughter is allergic to peanuts.`
- Low-salience: `Thanks!`

Result: the sleep job fetched two exchanges, promoted one, wrote `User's daughter has a peanut allergy.` into the brief, and marked the promoted exchange in SQLite.

What this proves:
- Raw episodic memory can now become compressed semantic memory.
- The sleep job has a real first path from conversation log to brief.
- Recall Lab can now test whether a brief created by consolidation changes later answers.

Next step: wire the Recall agent so every response reads the brief first, then compare it against the sliding-window baseline on the same five-turn memory failure.

---

## May 24, 2026. Brief-backed Recall agent beats the sliding-window failure.

Wired the experimental `RecallAgent` for the first time. On every turn it now loads the consolidated brief, composes working memory from brief plus recent turns, calls OpenRouter, writes the response to the episodic log, and keeps only a bounded recent-turn buffer.

Added `recall_demo.py` to compare two agents on the same five-turn sequence:
1. `My favorite color is blue.`
2. `I like reading history books.`
3. `I am testing memory systems today.`
4. `What is 2 + 2?`
5. `What is my favorite color?`

The sliding-window baseline used `window=2`. By turn 5, the favorite-color fact was outside the context window, and the model answered that it did not know.

The Recall agent used the same two-turn working window, but the sleep job promoted the favorite-color fact into the brief after turn 1. By turn 5, the agent answered: `Your favorite color is blue.`

Observed result:
- Sliding window final answer: failed to recall the favorite color.
- Recall Lab final answer: recalled blue from the consolidated brief.
- Brief after consolidation contained `User's favorite color is blue.` and `User enjoys reading history books.`

Read: this is the first concrete head-to-head where compressed semantic memory changes the final answer. It is still a tiny synthetic demo, not a benchmark result. But the full path now exists: raw exchange, salience score, promoted brief, brief-backed answer.

Next step: turn this into an evaluation harness with repeated scenarios, fixed expected answers, and measured success/failure instead of manual reading.

---

## May 24, 2026. v0.1 head-to-head eval harness.

Added `recall_lab.eval.v01_head_to_head`, the first repeatable eval harness for the May 29 lab notebook post. The harness runs the same five-turn favorite-color scenario through two agents:
- `sliding_window_2`
- `recall_lab_brief_window_2`

The expected answer for the final turn is `blue`. The scorer uses deterministic rules for v0.1: if the expected answer appears in the response, the turn is correct. If it does not appear and the model admits uncertainty, the failure mode is `honest_gap`. Otherwise the failure mode is `hallucinated`.

Observed run:
- Sliding window recall accuracy: `0.0` on 1 recall question. Failure mode: `honest_gap`.
- Recall Lab recall accuracy: `1.0` on 1 recall question. Failure mode: `correct`.
- Sliding output token estimate: `177`.
- Recall Lab output token estimate: `107`.

The token count is an approximation based on output text length, not billing data. The result is a tiny synthetic eval, not a benchmark. It is still useful because it is now repeatable and scored instead of read manually.

Next step: add four more scenarios so the June 4 v0.2 post can report five conversations instead of one.

---

## May 24, 2026. Configurable retail memory-week trial scaffold.

Added a configurable multi-day trial runner and the first retail scenario file. The scenario is stored as JSON, so the trial can change without touching core code:
- number of days
- day labels
- number of exchanges
- user messages
- final evaluation questions
- expected answers
- working-window size

The first scenario is `scenarios/retail_memory_week.json`. It simulates Amara, a returning retail customer, over four days and a final May 28 morning evaluation. The trial covers durable preferences, a safety-sensitive allergy, shipping-address correction, historical Lagos mentions, distractors, and final recall questions.

Added `recall_lab.eval.multiday_trial`, which can run sliding window, Recall Lab, or both against any compatible scenario file. It writes two outputs:
- `trial_result.json` for exact data
- `trial_report.md` for a screenshot-friendly lab report

Smoke test passed with a tiny one-day scenario.

Next step was to wire contradiction-aware memory updates into the sleep path before running the full retail trial, so Berlin can supersede Lagos instead of both facts simply living in the brief. This was completed in the next May 24 entry.

---

## May 24, 2026. Trace store, contradiction cap, and staged runner.

Implemented the remaining end-to-end memory path needed before the full retail trial:
- Added a JSONL memory trace store for promoted semantic memories.
- Added `TRACE_STORE_PATH` and `CONTRADICTION_COMPARE_LIMIT`.
- Added a `section` field to `MemoryTrace` so traces can render back into the brief.
- Wired sleep job through salience, trace creation, contradiction checks, validity transitions, activation ranking, and brief rendering.
- Capped contradiction checks to the top active traces by activation, default `3`, so full trials do not grow into unbounded pairwise comparisons.
- Added progress logs to the multi-day runner with `--verbose`.

Smoke checks passed:
- Trace store saved and loaded active/superseded memories.
- Brief rendering filtered out a superseded Lagos trace and kept active Berlin.
- Tiny one-day trial passed.
- Retail day-1 Recall-only staged run completed.

Retail day-1 staged result:
- 6 exchanges
- 4 promoted
- 4 active traces
- 2 skipped as low salience
- final eval accuracy after only day 1: `0.6`

Read: the system is now wired end to end, but full 4-day live runs should be staged because chat calls, salience judge calls, and contradiction calls are expensive. The safe ladder is day 1, then days 1-2, then full Recall-only, then both agents.


---

## May 24, 2026. Current repo state after trace-store integration.

The repo now tells one end-to-end story:

```text
episodic log
  -> salience judge
  -> memory trace store
  -> contradiction check
  -> validity state
  -> activation ranking
  -> consolidated brief
  -> RecallAgent answer
```

What is wired:
- raw exchanges persist in SQLite
- sleep job scores unpromoted exchanges
- high-salience exchanges become machine-readable traces
- new traces compare against the strongest active traces, capped at `3` by default
- corrections can supersede old traces
- active traces are ranked by activation before rendering into the brief
- the multi-day runner can run staged trials and write JSON plus markdown reports

What remains before the May 28/29 post:
- run the full four-day retail trial in stages
- compare Recall Lab against sliding window on the full scenario
- inspect the report for stale-memory behavior
- capture the report or trace table for the post

The system is structurally ready. The public result still depends on the full trial output.
---

## May 24, 2026. Full retail memory-week trial completed.

Ran the full four-day retail shopping assistant scenario with both agents:
- `sliding_window_2`
- `recall_lab_brief_window_2`

Scenario shape:
- 30 chat exchanges across May 24 night through May 27
- 5 final evaluation questions on May 28 morning
- working window: 2 turns

Result:
- Sliding window recall accuracy: `0.2`
- Recall Lab recall accuracy: `1.0`
- Sliding output token estimate: `5766`
- Recall Lab output token estimate: `2538`

Final eval answers:
- color preference: sliding passed, Recall Lab passed
- daughter's peanut restriction: sliding failed, Recall Lab passed
- current shipping city Berlin: sliding failed, Recall Lab passed
- historical shipping city Lagos: sliding failed, Recall Lab passed
- history book preference: sliding failed, Recall Lab passed

Sleep summaries:
- May 24 night: 4 promoted, 4 active traces
- May 25: 5 promoted, 1 corrected, 6 active traces, 7 total traces
- May 26: 2 promoted, 2 confirmed, 6 active traces
- May 27: 4 promoted, 2 confirmed, 8 active traces, 9 total traces

Most important trace result:
- `User (Amara) typically ships orders to Lagos.` became `superseded`.
- `User's shipping address is now Berlin; use Berlin for all future shipping.` stayed active and superseded the Lagos trace.
- A separate active historical memory preserved that Amara previously lived in Lagos and ordered goods there.

Read: this is the first full end-to-end Recall Lab win on a controlled multi-day scenario. The result is still synthetic and small, but it tests the full loop: episodic log, salience scoring, trace creation, contradiction, validity state, activation-ranked brief rendering, and final recall.

Report outputs:
- `reports/retail_memory_week/trial_result.json`
- `reports/retail_memory_week/trial_report.md`

Next step: inspect the generated report, capture a clean screenshot or table for the May 28/29 lab notebook post, and keep the claims scoped to one synthetic scenario.
---

## May 24, 2026. Scorer and brief-render fixes, full trial rerun.

Applied the first three review fixes before using the retail result publicly:

1. Fixed brief rendering. Trace sections now pass through the brief section aliases, so reports show human-readable headings like `Stable facts about the user` instead of raw keys like `stable_facts`.
2. Replaced the substring scorer for multi-day final evals with an LLM judge rubric. The scorer no longer marks a generic answer correct just because it contains the expected word.
3. Reran the full four-day retail trial with both agents.

Rescored result:
- Sliding window recall accuracy: `0.0`
- Recall Lab recall accuracy: `0.8`
- Sliding output token estimate: `5039`
- Recall Lab output token estimate: `2136`

The earlier sliding-window `0.2` was a false positive from substring scoring. The color answer mentioned blue as one generic option, not as remembered preference. The LLM scorer correctly marked it as hallucinated.

Recall Lab passed:
- favorite color: navy blue
- daughter allergy: peanut
- current shipping city: Berlin
- book preference: history books / African history

Recall Lab failed:
- historical shipping city: Lagos

Read: the corrected result is stronger and more honest. Recall Lab protects current truth after correction, but loses access to superseded history unless the user restates it later as history. That is now the main finding for the May 28/29 post.

Report outputs:
- `reports/retail_memory_week_rescored/trial_result.json`
- `reports/retail_memory_week_rescored/trial_report.md`

---

## May 24, 2026. Variance run, scorer false negatives, v3 ensemble judge.

Ran the four-day both-agents trial 5 times to see how stable one result is. Added `recall_lab/eval/variance.py` to run the campaign and aggregate the spread.

The v2 single-judge scorer reported:
- Sliding window: `0.00` every run.
- Recall Lab: `0.80` mean, `0.60` to `1.00` range.

Then I read all 25 graded Recall Lab answers by hand. The judge mislabeled 4 of them.

Receipt. Run 3 and run 4, same question, near-identical answers:
- Run 3: "You should prioritize navy blue since it's your favorite color." Scored hallucinated.
- Run 4: "You should prioritize navy blue since it's your favorite color!" Scored correct.

Same answer, opposite verdict. The judge runs at temperature 0 and still flipped. Three more: run 5 color, and runs 2 and 5 for the Lagos question, all named the correct fact and were scored hallucinated.

Hand-scored, the real result is 24 of 25 correct. One true miss: run 1, where the agent genuinely lacked the Lagos history and said so. That is an honest gap.

Diagnosis. The v2 scorer had two faults. A systematic bias: the rubric line "do not mark correct just because the expected word appears" made the judge suspicious of correct answers that name the fact. And residual noise: separate temperature-0 calls returned different verdicts on the same input.

Fix, v3 scorer in `metrics.py`:
- Sharpened the rubric with three worked examples. A confident answer that names the expected fact scores CORRECT. A generic list scores DRIFTED. A correct historical answer scores CORRECT.
- Added a majority vote over 3 judge calls. The rubric removes the bias. The vote removes the residual noise.
- Kept the substring scorer as the no-API-key fallback only.

Verification. Offline, the majority-vote logic passes on mocked verdicts. Live, the v3 scorer was run on the 4 answers v2 misjudged. All 4 now score CORRECT. The generic sliding-window answer still scores DRIFTED, so the fix did not bring back the old false positives. The run-1 honest gap still scores HONEST_GAP.

The scorer has now been through three versions:
- v1, substring match. Gave false positives. A generic answer that mentioned "cobalt blue" passed a color-recall question.
- v2, single LLM judge. Removed the false positives, added false negatives. It failed correct recalls as hallucinations and flipped between runs.
- v3, ensemble judge with a worked-example rubric. Removed both.

Trail kept. The v2 campaign is archived at `reports/variance/v2_single_judge/`. The v3 rerun writes to `reports/variance/v3_ensemble_judge/`.

Read: the variance run was meant to measure agent stability. It measured scorer stability instead. A single LLM-as-judge call is not a reliable oracle. The eval needs the same rigor as the system it grades.

Result, v3 rerun: pending. Run `python -m recall_lab.eval.variance`, then record the spread and the per-question pass counts here.

---

## May 24, 2026. Scorer default reconsidered: one call, with an audit.

The v3 fix above shipped a majority vote of 3 judge calls. On review that is the wrong default for a framework. Three calls on every grade reads as not trusting your own judge. It triples the cost. It treats the symptom.

So the design changed.

What is the real fix. The rewritten rubric, not the vote. When the v3 rubric was tested with a single call, all 4 answers the old judge had failed came back CORRECT, the generic list still scored DRIFTED, and the honest gap still scored HONEST_GAP. The prompt did the work. The vote did not.

Why one call can hold. A better prompt does not make the model deterministic. It makes the right answer obvious. The judge only flips on borderline cases, where a vague prompt left it unsure. A sharp rubric with worked examples moves cases out of that borderline zone.

The new design:
- `judge_failure_mode` defaults to one judge call. That is what ships.
- `judge_answer` returns the verdict plus the raw votes, so callers can audit.
- The variance runner scores with 3 calls on purpose. That is an audit, not the default. The summary now reports how often the 3 calls split.
- If the split rate is near zero, one call is confirmed safe and the audit can be retired. If it is not, the audit shows the vote earns its place.

The decision is not made by opinion. The variance run measures it.

The scorer arc for the post is now four steps: substring match, false positives. Single LLM judge, false negatives. Ensemble vote, correct but heavy. Sharper rubric with the vote demoted to a one-time audit.

Result, audited v3 rerun: pending. Run `python -m recall_lab.eval.variance`. Record the spread, the per-question pass counts, and the judge audit split rate here.
---

## May 25, 2026. Variance run and scorer audit.

Ran the audited variance campaign after demoting the scorer back to one call by default. The variance campaign still uses three scorer calls per final answer as an audit, then records whether the judge agreed with itself.

Result across five full retail trials:
- Sliding window accuracy: min `0.00`, max `0.00`, mean `0.00`
- Recall Lab accuracy: min `0.60`, max `1.00`, mean `0.88`

Per-question Recall Lab pass count:
- favorite color: `5/5`
- daughter peanut restriction: `5/5`
- current shipping city Berlin: `4/5`
- historical shipping city Lagos: `3/5`
- history books: `5/5`

Judge audit:
- 50 graded answers
- 1 split verdict
- split case: sliding-window answer to `What kind of books do I like?`, votes `drifted`, `honest_gap`, `honest_gap`

Read: the stronger scorer prompt made one-call grading mostly stable. The audit found a `2%` split rate, and the only split did not affect correctness because both labels were failures. The experimental result is now clearer: Recall Lab reliably beats the two-turn sliding window in this synthetic retail trial, but historical memory remains weaker than current truth and safety constraints.

Local reports:
- `reports/variance/v3_ensemble_judge/variance_summary.md`
- `reports/variance/v3_ensemble_judge/run_1` through `run_5`

