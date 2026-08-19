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

---

## May 25, 2026. Item 6: contradiction classifier moved to the strong model.

The audited variance run showed every Recall Lab miss sat on the Lagos-to-Berlin correction. Color, peanut, and books passed 5 of 5. Current city Berlin passed 4 of 5. Old city Lagos passed 3 of 5. The plain stored facts never failed. The correction did.

The correction is decided by `contradiction.classify()`. That call was running on `gpt-4o-mini`, the cheap agent model. The weak spot in the result and the weak model in the code were the same place.

What changed:
- Added `CONTRADICTION_MODEL` to `config.py`. It defaults to the judge model, `claude-sonnet-4.6`, and takes an env override, `RECALL_CONTRADICTION_MODEL`.
- `classify()` now uses it. Spotting an implicit correction is the hardest reasoning call in the system. The STALE paper logged on 2026-05-24 is built around that exact case. It belongs on the strong model.
- Dropped the `response_format` JSON-object parameter. It suited the cheap model. The strong model uses the system-message-plus-lenient-parse pattern that `judge.py` and `metrics.py` already use. Added `_extract_verdict_json` to handle a fenced or padded reply.
- Set the classifier to temperature zero. It was the last LLM call in the system still running above zero. The salience judge and the eval scorer were already there.

Verification: the offline contradiction checks pass. The live three-pair test on the strong model returns CORRECT, CONFIRM, UNRELATED, all correct, with clear reasons.

A side effect worth noting. With the classifier at temperature zero, the run-to-run spread no longer comes from any judge or classifier call. It comes from the agents answering above temperature zero, which is the realistic source. The variance docstring and summary text were corrected to say so.

What to expect. The strong classifier should lift the Berlin pass rate. Berlin is current truth: the agent says Berlin only when the sleep job correctly supersedes the Lagos trace, and that depends on `classify()` calling the pair CORRECT. Lagos is a different problem. A superseded trace is filtered out before the brief is built, so old history is only reachable when the user restates it. The model swap does not fix that. It is an architecture question for later.

Result: recorded in the next entry.

---

## May 25, 2026. v4 result: strong classifier, and the superseded-memory gap.

Ran the variance campaign on the strong contradiction classifier. Output in `reports/variance/v4_strong_classifier/`.

Result across five full retail trials:
- Sliding window accuracy: mean `0.00`, every run.
- Recall Lab accuracy: min `0.80`, max `1.00`, mean `0.88`.
- Range tightened from v3's `0.60` to `1.00`. The worst run improved.

Per-question Recall Lab pass count:
- favorite color: `5/5`
- daughter peanut restriction: `5/5`
- current shipping city Berlin: `5/5` (v3 was `4/5`)
- historical shipping city Lagos: `2/5` (v3 was `3/5`)
- history books: `5/5`

Judge audit: `0` split verdicts out of `50`. Every answer unanimous. The one-call scorer is confirmed.

Then I inspected the Lagos traces in all five runs. This is the part that matters.

- run 1: FAIL, honest_gap. Only trace present: "User typically ships orders to Lagos", superseded.
- run 2: PASS. Also has an active trace: "User previously lived in Lagos and frequently ordered goods there."
- run 3: FAIL, honest_gap. Only the superseded Lagos trace.
- run 4: FAIL, honest_gap. Only the superseded Lagos trace.
- run 5: PASS. Also has an active trace: "User previously lived in Lagos and currently lives in Berlin."

In all five runs the original "ships to Lagos" memory was superseded. The agent answered the history question correctly in exactly the two runs where the salience judge had also stored a separate, still-active "previously lived in Lagos" fact. In the other three only the superseded trace existed, and the agent honestly said it did not know.

Read: the strong classifier did its job. The Lagos trace was superseded in 5 of 5 runs, so supersession is now reliable, and Berlin is `5/5` because of it. Berlin across v3 and v4 is `9/10`.

The Lagos `3/5` to `2/5` move is not a classifier effect. It is judge-promotion luck. The classifier swap never touched the history path. The mean held at `0.88` only because Berlin gained a run and Lagos lost one. The mean hides the story; the per-question table is the story.

The real finding, now proven in the traces: Recall Lab has no retrieval path to a superseded memory. Once a fact is superseded, `current_traces()` filters it out and the brief never shows it. The agent recalls old history only when a different, still-active trace happens to record it. History recall is a side effect of the user repeating themselves. It is not a designed capability.

Next: give the agent a path to superseded memory. Render superseded traces into the brief under a separate past section, marked as history, never as current truth. Then rerun the campaign and watch Lagos.

Item 6 done. Classifier on the strong model, supersession reliable.

---

## May 25, 2026. Past section: superseded memory is retrievable again.

Built the fix for the superseded-memory gap from the v4 entry.

The cause, restated: a corrected fact moves to `superseded`, `current_traces()` keeps only active traces, and `render_brief()` rendered only those. The superseded fact stayed in the trace store but never reached the brief. The agent reads the brief, so it could not see history.

The change, three files:
- `brief.py`: added a sixth section, "Past, no longer current".
- `traces.py`, `render_brief`: it now also renders superseded traces, under the past section, each line prefixed "Previously:" so the present-tense wording is not read as current. Archived traces stay dropped.
- `working.py`: one prompt line. The past section is history, used only when the user asks about the past, never as a current fact.

No intent classifier. The agent reads the whole brief. A labeled current section and a labeled past section is enough for it to tell "now" from "before". A classifier would add an LLM call and a failure point for a job the answering model already does.

Verified offline: an active trace renders as a current fact, a superseded trace renders under the past section marked "Previously:", an archived trace is dropped, and there are no duplicate or raw-key headings.

Forgetting now removes authority, not the record.

Result, pending: rerun the variance campaign as v5. Lagos should climb from 2/5 toward 5/5. Berlin should hold at 5/5. If Berlin drops, the agent is reading the past section as current and the label needs strengthening.

---

## May 25, 2026. v5 partial, the credit wall, and a missing token cap.

Ran the v5 campaign on the past-section change. It stopped early.

What completed. Two of five runs finished before the wall:
- run 1: sliding `0.00`, Recall Lab `1.00`.
- run 2: sliding `0.00`, Recall Lab `1.00`.

Both completed runs passed all five questions, including the Lagos history question, `2/2`. In v4 that question was `2/5`. The past section is doing what it was meant to. Two runs is an early signal, not a result. The campaign needs the full five.

What stopped it. Runs 3, 4, and 5 failed with an OpenRouter HTTP 402: "This request requires more credits, or fewer max_tokens. You requested up to 65536 tokens, but can only afford 65447." Not a crash. The account ran low on credit.

The deeper cause. No LLM call in the repo set `max_tokens`. With no cap, OpenRouter pre-authorizes the model's full output budget, 65536 tokens for `claude-sonnet-4.6`, before every call. The real responses are a chat turn or a small JSON verdict, about 100 tokens. But the account has to hold credit for 65536. When the balance dropped below that hold, calls were rejected with credit still in the account.

The fix:
- Added `MAX_OUTPUT_TOKENS` to `config.py`, default `1024`, env override `RECALL_MAX_OUTPUT_TOKENS`.
- Passed it on all five `create()` calls: the agent, the sliding-window control, the salience judge, the eval scorer, and the contradiction classifier.

OpenRouter now pre-authorizes `1024` tokens per call, not `65536`. This does not lower the real spend. It removes a false wall and lets the balance run down to near zero before a 402.

Verified offline: all six changed files compile and import, and every `create()` call carries the cap.

Result, pending: add OpenRouter credit, then rerun the full v5 campaign and read the Lagos and Berlin pass rates.

---

## May 25, 2026. v5 result: the past section closes the history gap.

Full campaign completed, 5 of 5 runs. The token cap held: no 402, the credit wall is gone.

Result across five full retail trials:
- Sliding window accuracy: `0.00` every run.
- Recall Lab accuracy: `1.00` every run. 25 of 25 graded answers correct.

Per-question Recall Lab pass count:
- favorite color: `5/5`
- daughter peanut restriction: `5/5`
- current shipping city Berlin: `5/5`
- historical shipping city Lagos: `5/5` (v4 was `2/5`)
- history books: `5/5`

Judge audit: `0` split verdicts out of `50`. Three campaigns now agree the one-call scorer is stable.

Mechanism confirmed in the traces. Run 1's brief carries a section "## Past, no longer current" with the line "Previously: User (Amara) typically ships orders to Lagos." The agent answered the history question "You used to ship orders to Lagos." History recall now comes from the supersede chain rendered into the brief. In v4 it came from a redundant trace the salience judge minted by chance. v4 worked twice by luck. v5 works five times by design.

Berlin held at `5/5`. The agent told current truth from past truth off the labeled brief, with no intent classifier. The labeled-section design held.

Read, with the caveat. This is one scenario, five runs, a two-turn window that starves the baseline on purpose. 25 of 25 is a clean sweep on a narrow test, not a general claim. And the eval now has no headroom. A score of `1.00` means the retail scenario has nothing left to teach. The next experimental step is a harder scenario: multi-step corrections like Lagos to Berlin to Nairobi, longer gaps, and a question about the middle state.

The scorer and the two architecture fixes from this stretch all came from trace evidence, not guesswork. Forgetting removes authority, the record stays. The history question is answered by design.

---

## May 25, 2026. v6 relocation chain: two gaps in multi-step history.

Ran the harder scenario. `relocation_chain` is a two-step correction: Lagos to Berlin to Nairobi. Five runs.

Result:
- Sliding window: `0.00` every run.
- Recall Lab: mean `0.92`, runs `1.00, 0.80, 1.00, 1.00, 0.80`.
- Current city Nairobi: `5/5`.
- Previous city Berlin: `5/5`.
- First city Lagos: `3/5`.
- Color and shellfish controls: `5/5` each.
- Judge audit: `0` of `50` split.

The scenario did its job. It found two gaps, and they are not the same gap.

Gap A, ordering. Run 2 failed Lagos with the chain fully resolved. Both the Lagos and Berlin traces were superseded, and both rendered in the Past section. But the Past section is a flat list, sorted by activation. "Previously: Berlin" and "Previously: Lagos" carry no order. Asked for the first city, the agent could not tell which past city came first, and said it did not know. The activation sort floats the most recent past to the top, which is why "right before current" passes `5/5`. That is an implicit order. There is no explicit one, so "first ever" is an inference the agent makes only some of the time.

Gap B, chain resolution. Run 5 failed Lagos for a different reason. The two-step correction did not resolve. The Lagos trace ended up active, not superseded. The brief carried "User's shipping address is Lagos" and "User's current shipping address is Nairobi" as current facts at the same time. Lagos never reached the Past section. The likely cause is the contradiction compare limit. `classify()` checks each new trace against the top three active traces by activation. The salience judge mints several near-duplicate shipping traces. When the Nairobi correction arrived, a Lagos trace sat outside the compare window and was never superseded. The mechanism is worth confirming. The outcome is certain: a stale fact stayed marked current.

Read. Gap A is a rendering problem. The data is right and the order is missing. Gap B is a correctness problem. The data is wrong and a stale fact reads as current. Gap B comes first. Nairobi passed `5/5`, but in run 5 it passed despite a contradictory brief, because the corrections line is emphatic. Current truth is not solved yet. In run 5 it survived a broken memory state.

Fix directions:
- Gap A: render the Past section as an ordered lineage. Walk the supersede pointers, render oldest to newest, label the first city and the current one. Then "first city" and "before current" are both explicit.
- Gap B: a correction should supersede every active trace on the same topic, not only the top three by activation. Dedupe near-identical traces before the contradiction pass. The compare limit of three was a cost choice. It is now a correctness bug.

Also fixed: the variance summary title was hardcoded to `retail_memory_week`. It now reads the scenario name from the run.

Result, pending: fix Gap B, then Gap A, then rerun `relocation_chain` as v7.

---

## May 25, 2026. Fixing the two relocation-chain gaps.

Both gaps from the v6 entry are now fixed in code.

Gap B, chain resolution. The bug was in `_integrate_trace`. When a new trace arrived, it compared the trace against the top three active traces ranked by activation, and it superseded only the first contradicting trace it found, then returned.

Two faults in one function. A stale fact has low activation, so ranking by activation and keeping the top three drops exactly the traces a correction is meant to catch. And returning after the first correction leaves any other stale trace untouched. A two-step chain, Lagos to Berlin to Nairobi, needs the Nairobi correction to supersede every stale shipping trace at once.

The fix. `_integrate_trace` now compares the new trace against every active trace, and supersedes every one it corrects. The activation-ranked cap is removed. `CONTRADICTION_COMPARE_LIMIT` is gone from the config.

A cost note, because the cap was a cost guard. Comparing against every active trace also makes CONFIRM dedup reliable. A duplicate now always meets the trace it duplicates and gets dropped instead of stored. Fewer duplicates means a smaller active set, which means the all-traces scan stays cheap. The cap was suppressing dedup, which grew the active set, which is the cost it was meant to control. At real scale the right scoping is embedding similarity by topic, not an activation rank.

Gap A, ordered lineage. The Past section rendered superseded traces ranked by activation, so the order was implicit and recency-shaped. The fix renders them oldest first, sorted by `created_at`. The agent prompt now states the section is ordered oldest first, the first item earliest, the last item the most recent past. "First city" is the first entry. "City before current" is the last entry. Both are explicit.

Verification, offline. The 16 unit tests still pass. Three direct checks: one correction superseded both stale traces in a two-step chain, a duplicate trace was dropped and the original got a reference bump, and the Past section rendered Lagos before Berlin by creation time.

Result, pending: rerun `relocation_chain` as v7. Lagos should climb from `3/5`. Berlin and Nairobi should hold at `5/5`. The run-5 failure, a stale fact left active, should not recur.


---

## May 25, 2026. v8 patch: user-sourced salience and explicit past labels.

v7 exposed two failures that the v6 summary had hidden.

First, the salience judge was treating the agent answer as a source of user facts. The relocation scenario never has the user mention Lagos after day 1. When late Lagos traces appeared, they came from the agent recalling Lagos and the judge promoting that recalled answer as if the user had asserted it again. This is memory poisoning by the agent's own output.

Fix: the salience prompt now has a source rule. Durable facts may be extracted only from the user turn. The agent turn is context for judging intent, not a source of new facts about the user.

Second, the Past section still asked the agent to infer order from a flat list. Oldest-first helped the first-city question but hurt the right-before-current question. The list had one reliable slot, and moving the order traded one failure for another.

Fix: the Past section now labels the chain explicitly:
- `Earliest past: ...`
- `Past step N: ...`
- `Most recent past before current: ...`

The working-memory prompt now tells the agent how to use those labels: earliest for first-ever questions, most recent past for right-before-current questions.

Offline verification: 19 tests pass. Added a brief-render test that confirms a Lagos to Berlin to Nairobi chain renders Lagos as `Earliest past`, Berlin as `Most recent past before current`, and Nairobi as active current memory.

Prediction for v8: Nairobi should hold at 5/5. Berlin should recover from v7's 1/5. Lagos should stay high if the judge stops re-promoting agent-recalled history as new user truth.

---

## May 25, 2026. v8 result: source control fixed most of the chain, one lineage leak remains.

Ran `v8_user_sourced_lineage` on the relocation-chain scenario after two patches:
- the salience judge now extracts durable facts only from the user turn, not the agent turn.
- the Past section now labels lineage explicitly: earliest past, middle steps, most recent past before current.

Result across five runs:
- Sliding window: `0.00` mean.
- Recall Lab: `0.96` mean.
- Current city Nairobi: `5/5`.
- Right-before-current city Berlin: `4/5`.
- First city Lagos: `5/5`.
- Favorite color green: `5/5`.
- Daughter shellfish restriction: `5/5`.

Judge audit: one split out of 50, on a sliding-window failure. It did not affect Recall Lab scoring.

The patches worked. Compared with v7, Recall Lab moved from `0.80` mean to `0.96`, Lagos recovered from `4/5` to `5/5`, and Berlin recovered from `1/5` to `4/5`.

One failure remains. Run 1 still put a Lagos trace after Berlin in the Past section:
- Earliest past: Lagos
- Past step 2: Berlin
- Most recent past before current: Lagos

The agent then answered the right-before-current question with Lagos. The data was wrong, and the answer followed the data. This is no longer a prompt-order problem. It is a lineage construction problem: a stale Lagos trace can still enter the chain after Berlin, even with user-only salience. The next fix is not more prompt wording. The trace store needs topic-level lineage, so one shipping-address chain cannot accept an older city after a newer city unless the user explicitly reverts.

Read: v8 validates the source-control patch and explicit labels, but multi-step memory now needs lineage constraints. The chain is close, not solved.

---

## May 25, 2026. The run-1 leak was the soft patch, not a lineage gap.

A closer read of the v8 run-1 trace revised the entry above.

The v8 source-control patch was a prompt instruction. The judge saw the user turn and the agent turn, and was told to extract facts from the user turn only. The judge followed that most of the time. v8 at `0.96` is the proof. Run 1 is the proof it is not airtight.

Run 1's late Lagos trace carries `supersedes = 7`, the Berlin trace, so it was minted after Berlin and superseded it. The `relocation_chain` scenario never has the user mention Lagos after day 1. So the Lagos content came from an agent turn, and the judge extracted it despite the instruction. A prompt instruction is a request the model can decline.

That reframes the next fix. The lineage-constraint idea from the v8 entry is sound, but it is not the fix to reach for. A lineage constraint only does work when a bad trace is created. With airtight source control every trace is user-asserted, so every change to the chain is a real user statement, and the constraint never fires. A constraint also needs to tell a real revert from a phantom, which is the same user-versus-agent question source control already answers. Source control is the foundation.

The fix. The judge no longer sees the agent turn at all. `JUDGE_PROMPT` shows only the user turn, and `score_exchange` passes only `exchange.user`. The judge cannot extract a fact it cannot see. The soft rule is now structural.

The cost is small. The judge loses the agent turn as scoring context. Durable user facts, preferences, allergies, addresses, are self-contained in the user turn. A pure confirmation like "yes please" scores low and is skipped anyway.

Verified offline: judge.py compiles, the agent placeholder is gone from the prompt, `format(user=...)` works without a missing key, and the 19 tests pass.

Result, pending: rerun `relocation_chain` as v9. The run-1 leak should not recur. Berlin should reach `5/5`. If a clean run still corrupts the chain through a different path, a user turn the judge mis-reads for example, that is when the lineage constraint earns its place.

---

## May 26, 2026. v9 result: user-only salience closes the relocation chain.

Ran the relocation-chain variance campaign after making the salience judge structurally user-only.

Command:

```bash
python -m recall_lab.eval.variance --scenario scenarios/relocation_chain.json --label v9_user_only_judge
```

Result across five runs:
- Sliding window: `0.00` mean, every run.
- Recall Lab: `1.00` mean, every run.
- Current city Nairobi: `5/5`.
- Right-before-current city Berlin: `5/5`.
- First city Lagos: `5/5`.
- Favorite color green: `5/5`.
- Daughter shellfish restriction: `5/5`.
- Judge audit: `0` split verdicts out of `50` graded answers.

The run-1 leak from v8 did not recur. The judge cannot promote agent-generated history if the agent turn is not in its prompt.

Trace read: the chain is clean. Lagos is superseded. Berlin is superseded. Nairobi remains active. The Past section renders a two-entry lineage: earliest past Lagos, most recent past before current Berlin. The agent uses those labels correctly.

Read. The structural source boundary fixed the memory-poisoning bug. The system no longer turns its own recalled answer into fresh user memory.

This also changes the lineage-constraint question. A topic-level lineage constraint is still a plausible future backstop, but v9 shows it is not needed for the failure observed in v8. The root problem was source leakage, not lineage policy.

Current public claim, scoped tightly:

On two synthetic multi-day scenarios, a brief-backed memory system with validity state, a Past section, and user-only salience beats a two-turn sliding-window baseline. This is a mechanism result, not a benchmark.

What remains before stronger claims:
- vector retrieval control
- long-context control
- larger scenario set
- equal-token-budget baseline
- human-curated and random-curated brief controls
- decay policy

The next build should not make the current scenario easier. Both retail v5 and relocation v9 are maxed at `1.00`. The next useful experiment needs a harder scenario or a new control.

---

## May 26, 2026. Documentation sync after v9.

Updated `README.md` to match the actual system state.

The README now records:
- user-only salience judge
- `Past, no longer current` brief section
- explicit past-lineage labels
- retail v5 result
- relocation-chain v9 result
- judge audit status
- current incomplete controls
- the scoped public claim

This matters because the repo had drifted. It still said the full four-day retail trial and sliding-window comparison were incomplete, even though both the retail and relocation-chain campaigns had completed. The README now tells the same story as the code and the research log.

---

## May 29, 2026. Vector-retrieval control and equal-token-budget runner.

Built the two controls the v9 entry named as the next useful work: a fair-budget baseline and a vector-retrieval baseline. Both attack the same objection. Every result so far beats a deliberately starved two-turn window, so a skeptic can say Recall Lab only wins because the baseline was fed less.

What was added:
- `controls/vector.py`: the flat vector-retrieval control. Each completed exchange is embedded into an isolated in-memory ChromaDB collection; each turn retrieves the top-k most similar exchanges and prepends them. This is the Mem0 / standard-RAG default. It has no validity state by design: a superseded fact (Lagos) and the correction that replaced it (Berlin) are both just documents, and whichever scores more similar to the question wins regardless of which is still true. That is the failure this control should expose. The embedding function and client are injectable so tests run offline without a key.
- `controls/budgeted.py`: a sliding window bounded by an input-token budget instead of a fixed turn count. It keeps the most recent turns that fit under the budget.
- `eval/equal_budget_trial.py`: runs the Recall agent, measures its mean input tokens per chat turn, then runs the budgeted recency baseline at that same budget (scalable with `--budget-scale`) on the same scenario. Reports accuracy at equal token budget and writes a read that states plainly whether the gap survives, ties, or reverses.
- Input-token tracking: every agent now exposes `last_input_tokens`, and `multiday_trial` records an input-token estimate per turn. The trial report gained a mean-input-tokens column, and `--agents` gained `vector` and `all`.
- Tests: `test_vector_control.py` pins the retrieval store (top-k, isolation, empty-store, key guard) with a deterministic offline embedding function. `test_budgeted_window.py` pins context selection (drops oldest first, never overflows the budget). All 30 tests pass; ruff clean.

What this does not yet show: results. The runners are implemented and unit-tested, but no live campaign has been run. The model call path needs OPENROUTER_API_KEY, same as the other agents, so the offline tests cover plumbing only.

Next step: run both live campaigns on `relocation_chain`. Two predictions to check.
1. Vector control. It should pass the current city and stable facts, and fail or wobble on the right-before-current city, because retrieval has no way to order superseded facts. If it passes everything, the validity machinery is doing less than claimed and that needs explaining.
2. Equal budget. The budgeted window at `1.0x`, and ideally at `1.5x`, should still lose to Recall Lab. If a larger raw-recency budget closes the gap, the selective-consolidation claim is weakened for this setup, which is the falsification the protocol asks to surface.

Whatever the live runs show, write it here before touching the README claim.

---

## May 29, 2026. Live results: vector control and equal-token-budget runs.

Ran the two new controls live on `relocation_chain`. Single runs, not variance campaigns, so read the spread caveat at the end before quoting any single number.

### Vector-retrieval control, full four-day, all three agents

| Agent | Accuracy | Shape |
| --- | --- | --- |
| sliding_window_2 | 0.00 | every answer an honest gap |
| vector_topk_5 | 0.40 | passed the stable facts, failed the whole chain |
| recall_lab_brief_window_2 | 1.00 | all five correct |

The vector control did exactly what the May 29 build entry predicted, and the failure is sharper than expected. It passed the two un-superseded facts, green and shellfish. It failed every superseded one. Asked the current city it answered Berlin, the stale middle state. Asked the city right before the current one it answered Nairobi, the current state. It inverted the order of the relocation chain. Asked the first city it said it did not know. With no validity state, Lagos, Berlin, and Nairobi are three similar documents, and cosine similarity cannot say which is current or what order they arrived in. This is the failure the control exists to expose, and it is the clearest single illustration so far of why rank without validity is not enough. It echoes the May 20 activation finding at the retrieval layer: similarity, like activation, ranks strength, not truth.

### Equal-token-budget control

Recall Lab's measured cost was about 440 to 454 input tokens per turn. The budget-bounded sliding window was given that budget, then 1.5x of it.

| Budget | recall_lab | budgeted recency baseline |
| --- | --- | --- |
| 1.0x (454 tok) | 0.60 | 0.00 |
| 1.5x (660 tok) | 1.00 | 0.00 |

The recency baseline scored 0.00 at both budgets. At 1.5x its mean usage was 398.6 tokens, under the 660 cap, so it had headroom and still failed. The reason is structural. A fixed budget of recent raw turns cannot hold a fact stated on day 1 once 25 later turns have pushed past it, no matter how the budget is spent. Recall Lab compresses that fact into a brief that persists. So the win is not prompt length. Handing recency an equal or larger budget does not close the gap; it leaves it at zero. The protocol's equal-token-budget objection is answered for this scenario.

### The honest caveat

These are single runs. Recall Lab scored 1.00 in the all-agents run and the 1.5x run, but 0.60 in the 1.0x run, where it dropped both chain-order questions and said the first city was Berlin. That is the same above-zero-temperature variance the variance runner exists to measure. The baseline result is robust: 0.00 is 0.00 whether the comparison point is 0.6 or 1.0. The size of Recall Lab's lead is not yet a stable number from these runs.

### Next step

Run both controls through `eval/variance.py` so the lead is a mean over five runs with a judge audit, the same bar the retail v5 and relocation v9 claims already meet. Only after that should the README's public claim be widened to include the vector and equal-budget controls. Until then the claim stays as scoped on May 26.

---

## May 29, 2026. Variance campaigns: the lead is stable across runs.

Ran both new controls through `eval/variance.py` with the 3-call judge audit, the same bar v5 and v9 met. This is the answer to the single-run caveat in the entry above. First batch was lost to a transient connection blip; hardened the OpenRouter client with retries (see below) and reran.

### Vector-retrieval control, v10, 4 completed runs

Run 4 was dropped by a provider-side content-filter false-positive, not a code fault. See the provider-noise note below. Four runs completed.

| Agent | min | max | mean |
| --- | --- | --- | --- |
| sliding_window_2 | 0.00 | 0.00 | 0.00 |
| vector_topk_5 | 0.40 | 0.80 | 0.55 |
| recall_lab_brief_window_2 | 1.00 | 1.00 | 1.00 |

Per-question, across the 4 runs:
- vector passed favorite color 4/4 and the shellfish restriction 4/4, the two facts that are never superseded.
- vector passed current city 1/4, right-before-current city 2/4, first city 0/4. The chain is where it fails.
- recall_lab passed all five questions 4/4.

The single-run 0.40 for the vector control was the low end of a 0.40 to 0.80 spread. Even at its best run it never solved the first-city question. The pattern holds: similarity retrieval keeps stable facts and cannot order superseded ones.

### Equal-token-budget control, v10, 5 completed runs

| Agent | min | max | mean |
| --- | --- | --- | --- |
| recall_lab_brief_window_2 | 1.00 | 1.00 | 1.00 |
| budgeted_window | 0.00 | 0.20 | 0.04 |

The budget-matched recency baseline averaged 0.04. It passed one question in one run, the shellfish restriction, and nothing else. Giving recency Recall Lab's full per-turn token budget does not close the gap. The equal-budget objection is now answered with a spread, not a single draw.

### The single-run dip did not recur

The May 29 results entry flagged that recall_lab scored 0.60 in the 1.0x single run. Across these nine completed variance runs, four vector and five equal-budget, recall_lab scored 1.00 every time. The 0.60 was an unlucky single draw, plausibly worsened by provider-routing noise. The stable read on `relocation_chain` is: recall_lab 1.00, sliding 0.00, vector ~0.55, equal-budget recency ~0.04.

### Judge audit

Zero split verdicts. 0 of 60 graded answers in the vector campaign, 0 of 50 in the equal-budget campaign. The judge agreed with itself on every answer, consistent with v9. One judge call remains enough.

### Client hardening

The first rerun lost nine of ten runs to a transient `APIConnectionError`. Each agent built its own OpenAI client inline with the default two retries, which a multi-minute blip blows past. Added `recall_lab/llm.py` with a single `chat_client()` factory at six retries and a 60s timeout, and routed all seven call sites through it: the two new controls, sliding, the recall agent, the eval judge, the salience judge, and the contradiction classifier. The rerun saw no connection errors.

### Provider-routing noise, worth fixing before bigger runs

One run died on an Azure content-filter false-positive: a benign shopping-assistant prompt about cities and a shellfish allergy was flagged as a jailbreak. OpenRouter routes `openai/gpt-4o-mini` across providers non-deterministically, so the model behind a given call varies run to run. That adds variance unrelated to the memory architecture and can kill a run outright. Before scaling to the 30-conversation protocol, pin the provider, for example via OpenRouter provider preferences, so the experiment measures memory strategy and not provider lottery.

### Status of the public claim

These results clear the variance bar v5 and v9 met, so the README's scoped claim can now widen to name the vector and equal-budget controls. Caveats that stay attached: a single scenario, four to five runs, no statistical test yet, and the provider-routing noise above. The 30-conversation protocol in `protocol.md` is still the next real milestone.

---

## May 31, 2026. v11 pinned reruns: the spread collapsed.

After PR #14 (agent pin to OpenAI) and PR #15 (judge pin to Anthropic) merged, reran the v10 campaigns on `relocation_chain` under the pinned setup. Two campaigns, 5 runs each, 3-call judge audit. Both completed 5/5. Labels `v11_pinned_vector_control` and `v11_pinned_equal_budget_1x`.

Pre-pin v10 vs pinned v11, mean recall accuracy:

| Agent | v10 (pre-pin) | v11 (pinned) |
| --- | --- | --- |
| sliding_window_2 | 0.00 | 0.00 |
| vector_topk_5 | 0.55 (range 0.40-0.80, n=4) | 0.40 (range 0.40-0.40, n=5) |
| budgeted_window (equal budget) | 0.04 (range 0.00-0.20) | 0.00 (range 0.00-0.00) |
| recall_lab_brief_window_2 | 1.00 | 1.00 |

The read. The spread tightened to zero on both controls. Pre-pin, the vector control ranged 0.40 to 0.80; pinned, it is 0.40 every run. Pre-pin, the budgeted recency baseline wobbled 0.00 to 0.20; pinned, it is 0.00 every run. Recall Lab held 1.00 throughout, pre-pin and pinned.

This is the second of the two useful outcomes named when the rerun was planned. The numbers did not just hold within their old intervals; the variance that produced those intervals was largely provider-routing noise. With the model fixed to one provider per family, the controls are deterministic on this scenario at temperature zero for the judge and near-deterministic for the agents. So the pin is not only a reproducibility hygiene step. It is evidence that part of the pre-pin spread came from the provider lottery, not from the memory strategy.

Per-question, pinned vector still tells the same mechanism story: color 5/5 and shellfish 5/5, the two facts that never change, and 0/5 on all three chain questions (current city, previous city, first city). Similarity keeps stable facts and cannot order superseded ones. The pin sharpened the picture: the chain failure is now total and consistent, not occasionally masked by a lucky retrieval.

Judge audit: vector campaign 1 split verdict of 75 graded answers (run 1, the vector previous-city answer split correct/hallucinated/hallucinated, majority hallucinated, scored as a miss). Equal-budget campaign 0 splits of 50. Consistent with v9 and v10: the judge is stable.

Headline table from here on cites pinned runs. The pinned numbers are: sliding 0.00, equal-budget recency 0.00, vector 0.40, Recall Lab 1.00, all on `relocation_chain`.

Operational note. The reruns took several attempts. Session-bound background runs were killed by session interrupts, and one earlier batch lost runs to a transient APIConnectionError. The clean 5/5 came from running both campaigns directly in a terminal, sequentially. For the 30-conversation campaign, run it in a real terminal or a detached process, not a session-tied background job.

---

## May 31, 2026. Episodic read-time-judge baseline, and the paper that motivates it.

Paper of note: "Useful Memories Become Faulty When Continuously Updated by LLMs" (arxiv 2605.12978). It tested repeated LLM rewriting of memory and found the memory degrades over time. Its winning recipe was to keep raw traces and decide the current answer at read time, which beat the rewriting approaches.

This cuts toward Recall Lab's core premise, so it is worth confronting directly. The sleep job is a rewriting loop: it compresses, supersedes, and re-renders the brief each day, which is the pattern the paper punishes. The protection is that the raw episodic log is never discarded; the brief is derived and the SQLite trace store is ground truth. The paper's winning recipe is half of what the system already keeps.

So built the control the paper implies: `recall_lab/controls/episodic.py`, the `EpisodicJudgeAgent`. Keep every statement verbatim, inject the whole log each turn, ask the model to work out the current answer at read time. No compression, no supersede, no consolidation. It implements the `.respond` protocol and reports `last_input_tokens` so the runner can chart its growing input bill against the brief's flat one.

Wired into `multiday_trial.py` as `--agents episodic` (and folded into `all`) with `run_episodic_trial`, and into `variance.py`'s lineup. Tests in `tests/test_episodic_control.py` cover the raw-history plumbing offline: verbatim retention, oldest-first order, nothing dropped or compressed, and the API-key guard. 41 tests pass, ruff clean.

What it answers. Does validity-state consolidation actually beat keeping everything raw? Two ways the brief can still win. Accuracy: if raw history confuses the model on a long correction chain, the brief's explicit Past section wins on correctness. Cost: even if raw ties on accuracy on the short relocation chain, it pays a growing input-token bill while the brief stays bounded. The crossover, where long logs make raw too expensive and consolidation starts to pay, is the Chapter 3 result.

Next step. Run `--agents episodic` on the relocation chain under the pinned provider, alongside the existing controls, and chart accuracy and mean input tokens per turn. If raw beats the brief on accuracy here, the paper is right for this setup and the sleep job needs to justify itself on cost or on a longer scenario. Report either way before widening any claim. Not run yet; the runner and control are in place.

Other papers from the same scan, logged for the trail: Memora: From Recall to Forgetting (arxiv 2604.20006), EvoMemBench (arxiv 2605.18421), Memory-Induced Tool-Drift in LLM Agents (arxiv 2605.24941).

---

## June 7, 2026. Strong RAG control built, and the Chapter 3 campaign pre-registered.

The Chapter 3 claim is that retrieval misses authority. The risk is that it has only been tested against flat top-k vector search, which no serious team ships. Beating a strawman is not a result. So I built the strong end of retrieval as a control and pre-registered the predictions here before running, so the result cannot be reverse-fit to the chapter.

### What was built

`recall_lab/controls/strong_rag.py`, `StrongRAGAgent`. The industry-standard stack on the same `.respond` protocol as the other controls:

- Query rewriting. The raw turn is rewritten into an explicit retrieval query, so "where do you ship today" becomes a query for the current shipping city.
- Hybrid retrieval. Dense vector search (ChromaDB) and lexical search (BM25, with a token-overlap fallback) fused with Reciprocal Rank Fusion (k=60).
- Recency boost. Each exchange carries a turn index; a tunable boost lifts newer exchanges of equal relevance. This is the heuristic that could let it prefer Berlin over Lagos without any validity state.
- Reranking. The fused candidates are reranked before the top-k context is composed. LLM reranker by default, injectable for a cross-encoder.

Still no validity state by design. Any win on the relocation chain comes from recency, any failure isolates the authority gap. Every external piece is injectable, so the fusion, recency, and rerank logic are covered by `tests/test_strong_rag.py` offline, 9 tests, no network or API key. Wired into `multiday_trial.py` and `variance.py` as `--agents strong_rag` and folded into `all`.

### Pre-registered predictions (relocation_chain, pinned provider, 5 seeds)

Written before the campaign runs. The point is to commit, then report against this, win or lose.

1. Strong RAG clears the stable facts: favorite color and shellfish allergy at or near 5/5, same as flat vector.
2. Strong RAG beats flat vector on the chain. Flat vector is 0/5 on current, previous, and first city (pinned v11). I expect the recency boost to recover the current city most runs, so current-city accuracy rises above zero.
3. Strong RAG still fails the ordered chain. Previous-city and first-city stay low, because recency ranks by newness and the chain needs the order of supersession, which recency flattens. Mean on the three chain questions stays well below Recall Lab's 1.00.
4. Raw episodic read-time judge ties or beats strong RAG on accuracy here, because the whole log fits and the model can reason over it, but pays a growing input-token bill that strong RAG and the brief do not.
5. Recall Lab brief holds 1.00 on the chain at a bounded token cost.

If prediction 3 is wrong and strong RAG also reaches 1.00 on the ordered chain, the validity claim narrows honestly: engineered retrieval is enough for this scenario, and the case for validity state moves to a harder scenario (an adversarial re-assertion of an old fact, or a longer chain). Report either way.

### Next step

Run `scripts/run_chapter3.sh` from the repo root on a real terminal. The key is read from `.env` by `config.py`, same as every campaign. It runs the full lineup (`--agents all`) plus the equal-budget control across 5 seeds, under the pinned provider, with the 3-call judge audit. Then fold the headline table into the Chapter 3 draft, replacing the "what I have not tested yet" section with numbers. Not run yet; the control, runner, and predictions are in place.

---

## June 7, 2026. v12 Chapter 3 results: strong RAG climbs the recent links and fails the oldest.

Ran `scripts/run_chapter3.sh` on `relocation_chain`, 5 seeds per agent, pinned provider, 3-call judge audit. Zero split verdicts: 0 of 125 in the lineup, 0 of 50 in the equal-budget pass.

Headline, mean recall accuracy and mean chat input tokens per turn over 5 runs:

| Agent | Accuracy | Mean chat input tokens |
| --- | --- | --- |
| sliding_window_2 | 0.00 | 267 |
| vector_topk_5 | 0.52 | 373 |
| strong_rag | 0.76 | 479 |
| episodic_judge | 1.00 | 974 |
| recall_lab_brief_window_2 | 1.00 | 438 |
| budgeted_window (equal-budget) | 0.08 | matched |

Per-question, strong_rag: current city 4/5, previous city 5/5, first city 0/5, favorite color 5/5, shellfish 5/5.

### Predictions vs results (against the June 7 pre-registration)

1. Stable facts at or near 5/5: confirmed. Color and shellfish 5/5 for vector, strong RAG, episodic, and the brief.
2. Strong RAG beats flat vector and recovers the current city: confirmed. 0.76 vs 0.52, current city 4/5 vs 0/5.
3. Strong RAG still fails the ordered chain: half confirmed. It failed the first (oldest) city 0/5, as predicted, but it recovered the previous city 5/5, which I predicted would stay low. The failure is narrower and deeper than I guessed: recency recovers the recent links and collapses only on the oldest superseded fact.
4. Episodic ties on accuracy at a higher, growing token cost: confirmed. 1.00 at 974 tokens/turn vs the brief's 1.00 at 438, and the episodic bill grows with the log.
5. Recall Lab brief 1.00 at bounded cost: confirmed.

### Reads

- The "RAG misses authority" claim survives, sharpened. A full industry stack (query rewrite, hybrid, RRF, recency, rerank) recovers the recent end of a correction chain and fails the oldest link, because recency approximates authority near the present and decays into the past.
- On this short chain, validity state does not beat raw retention on accuracy; both hit 1.00. The validity win here is cost: 438 vs 974 mean input tokens, and the gap widens with conversation length. This is the cost crossover the May 31 episodic entry called "the Chapter 3 result." An accuracy separation between brief and raw needs a longer or adversarial scenario, which is Chapter 4 and 5 work.
- Claim 3 in research-claims.md (retrieval finds candidates, authority decides which is current) now has direct strong-RAG evidence and is a promotion candidate to `tested`, pending the public post landing. Promotion is Cynthia's call per that file's rule.

Folded into Chapter 3 (`posts/chapters/03-rag-misses.md`, draft v2). The full per-question table and this scorecard are the spine of the follow-up lab note.

---

## June 7, 2026. v13 pre-registration: date-metadata filtering for strong RAG.

Cynthia asked whether v12 strong RAG did metadata filtering by date added. It did not. v12 used a recency boost on insertion order, with no stored date and no filter. To close the "industry-standard RAG" claim against that exact objection, I added `strong_rag_dated`: real `added_at` timestamps stored as chunk metadata and sourced from the scenario dates (set per day via `set_clock`), timestamp-based recency, and an explicit metadata filter by date added (`recency_window_days`, default 1.5 days from config). It runs alongside the turn-order `strong_rag` in `--agents all`, so the v13 campaign produces both side by side.

Predictions on relocation_chain, 5 seeds, pinned provider, written before the run:

1. `strong_rag_dated` matches `strong_rag` on the stable facts: color and shellfish at or near 5/5.
2. On the current city, dated is at least as good as turn-order. Both the date recency and the 1.5-day window favor the most recent fact.
3. On the first (oldest) city, dated stays at or near 0/5, and the date filter can make the previous-city question worse than turn-order, because a 1.5-day window drops the two oldest cities before reranking. Metadata-by-date filtering helps the present and cannot recover a fact older than the window.
4. Net: date-metadata filtering relocates the gap toward the present. It does not close it. The validity brief still holds 1.00 on the full chain.

If `strong_rag_dated` reaches 1.00 on the ordered chain, the validity claim narrows to "needs a harder scenario" and I report it. Run with `scripts/run_chapter3.sh` (label v13, key from `.env`). Not run yet; the control, config knob, runner wiring, and predictions are in place, and the offline tests pass.

---

## June 7, 2026. v13 results: date-metadata filtering makes strong RAG worse, and deletes a stable fact.

Ran the v13 campaign, then reran the lineup. The equal-budget pass completed 5/5 (brief 1.00, budgeted recency 0.00). The lineup pass completed 4 of 5 runs (run 5 dropped both times, likely an API blip mid-batch), so the numbers below are n=4. The spreads are near zero, so n=4 is reliable here; chasing the 5th run is optional.

Mean recall accuracy and mean chat input tokens per turn (n=4):

| Agent | Accuracy | Mean chat input tokens |
| --- | --- | --- |
| sliding_window_2 | 0.00 | 268 |
| vector_topk_5 | 0.50 | 350 |
| strong_rag (turn recency) | 0.80 | 473 |
| strong_rag_dated (date-metadata filter) | 0.40 | 477 |
| episodic_judge | 1.00 | 993 |
| recall_lab_brief_window_2 | 1.00 | 444 |

Per-question, strong_rag (turn recency): current city 4/4, previous city 4/4, first city 0/4, color 4/4, shellfish 4/4. It fails only the oldest link.

Per-question, strong_rag_dated (date filter): current city 4/4, previous city 0/4, first city 0/4, color 4/4, shellfish allergy 0/4. It perfected the present and deleted every fact older than the window.

Judge audit: 1 split of 120 graded answers (run 3 vector previous-city, majority correct). Stable.

### Predictions vs results (against the June 7 v13 pre-registration)

1. Dated matches the stable facts: WRONG, and this is the interesting miss. Color held at 2/2, but the shellfish allergy dropped to 0/2. The allergy was stated on day 1 and never changed, yet the 1.5-day date window filtered it out for being old. A hard date filter cannot distinguish "old but still true" from "old and superseded," so it deletes both.
2. On the current city, dated is at least as good as turn-order: CONFIRMED. Dated hit 2/2, the only retrieval agent to nail the current shipping city every run. The filter is very good at the present.
3. First city stays near 0 and previous-city can get worse: CONFIRMED and stronger. Previous city fell from strong_rag's 4/4 to 0/4 under the filter, and first city stayed 0/4.
4. Net: date-metadata filtering relocates the gap toward the present, does not close it, and the validity brief still holds 1.00: CONFIRMED. Dated 0.40 < turn-recency strong_rag 0.80 < brief 1.00, and the brief does it at a lower token cost (444 vs the dated agent's 477 and episodic's 993).

### Read

The headline for the Jun 16 lab note: bolting the most-cited "industry standard" date filter onto strong RAG did not help, it dropped overall accuracy from 0.80 to 0.40. It bought a perfect current-city answer by deleting every fact older than the window, including a stable allergy that was always true. This is the cleanest demonstration in the lab so far that recency and date heuristics are proxies for authority, not authority. Authority has to know that a fact was superseded, not merely that it is old. That is what the validity brief encodes and what no retrieval filter can infer from a timestamp.

This strengthens research-claims.md Claim 3 (retrieval finds candidates, authority decides which is current). It is now a strong promotion candidate to `tested`. Promotion is Cynthia's call.

Next: the lineup is at n=4 with near-zero spread, enough to draft the Jun 16 lab note around the date-filter result. Chasing a clean n=5 is optional. Chapter 3 keeps its v12 5-seed numbers (strong_rag 0.76) and does not need the dated result; the dated finding belongs to the lab note.

---

## June 9, 2026. v14 pre-registration: fair RAG sweep, removing the knob-bias question.

Cynthia raised the right methodological worry: the v12 strong-RAG result used recency_weight=0.30 and top_k=5, and the first-city miss was partly a coverage effect (only 5 snippets in context, recency pushed the oldest city out, the model abstained). So the per-question numbers could be an artifact of two choices. The fair sweep removes that question. New runner `recall_lab/eval/fair_rag_sweep.py`, run via `scripts/run_fair_rag.sh`.

Added to `StrongRAGAgent`: `show_timestamps`, which prefixes each retrieved snippet with its real date so the model can reconstruct order if the information is in front of it. `run_strong_rag_trial` now passes recency_weight, top_k, candidate_k, and show_timestamps. Offline tests cover the timestamp rendering (16 tests pass).

Sweep grid on relocation_chain: recency ablation rw=0.0 / 0.3 / 0.6 at top_k=5; top_k sweep k=5 / 10 / full at rw=0.3; fair shot at k=full with timestamps visible. Reference baselines (vector 0.52, episodic 1.00 ~974 tok, brief 1.00 ~438 tok) are cited from v12, not re-run.

Predictions, written before the run:

1. Recency ablation at k=5: the first city stays at or near 0 for rw=0.0, 0.3, and 0.6. Recency is not what causes the first-city failure; with only 5 snippets the oldest fact rarely makes the cut. If rw=0.0 recovers the first city, then recency was the cause and I report that.
2. top_k sweep: as k grows, the oldest city gets retrieved more often, so first-city accuracy rises with k. The failure at small k is coverage, not an inability to read a retrieved fact.
3. Fair shot (k=full + timestamps): the model sees every city with its date, so it should reconstruct the order and approach 1.00. This config is effectively the keep-everything agent with timestamps, so its mean chat input tokens should land near episodic's ~900 and well above the brief's ~438.
4. Net honest read: RAG can order the chain only by carrying the whole history (high k), which costs unbounded tokens. Bounded-context RAG (small k) cannot, at any recency setting. The validity brief's value is bounded cost at full accuracy, not unique accuracy.

If the fair shot does not reach ~1.00 even with everything visible and dated, that is a stronger result (the retrieval framing itself fails to order a chain) and I report it. Run with `RUNS=3 scripts/run_fair_rag.sh` first for cost, then 5 for final.

### Addendum, June 9: 1-seed meter run + a fairness fix before the full cycle

Ran a 1-seed meter pass ($0.11 for the 6 configs, so a 5-seed final is about $0.55 to $0.70). Read the traces before scaling. Three things:

- Query rewrites are sane, so the pipeline is sound, not buggy.
- The first city failed in every config at n=1, including kFULL and fair-shot, which is the de-bias signal: the failure is not an artifact of recency weight or top_k.
- New and important: at full context (30+ snippets), the model gets distracted. On both the current and the first city it sometimes answers a different question entirely ("your favorite color is green") or returns the stale city (Berlin). So more retrieval did not help; it introduced lost-in-the-middle distraction. Lagos was present in context and the model still failed it.

That exposed a fairness gap. The full-context configs presented snippets in relevance order, which scrambles chronology. A careful production system answering a temporal question would sort retrieved memory by time. So I added a `chronological` option and a 7th config, `strong_fairshot_chrono`: full context, timestamps visible, snippets ordered oldest-first. Prediction: if even time-ordered full context fails the first city, the result is bulletproof; if chronological ordering rescues it, the honest finding becomes "you must reconstruct order explicitly before answering," which is the same authority argument from the other side. `rank-bm25` is now a dependency, so the final run uses real BM25 hybrid retrieval, not the token-overlap fallback. 17 offline tests pass. Run the full cycle at `RUNS=5`.

### v14 results, June 9 (5 seeds, real BM25; a couple of configs landed 4 runs)

| Config | mean acc | first-city | mean chat tokens |
| --- | --- | --- | --- |
| strong_rw0.0_k5 | 0.60 | 0/5 | 440 |
| strong_rw0.3_k5 | 0.68 | 0/5 | 429 |
| strong_rw0.6_k5 | 0.75 | 0/4 | 406 |
| strong_rw0.3_k10 | 0.80 | 0/4 | 628 |
| strong_rw0.3_kFULL | 0.76 | 0/5 | 902 |
| strong_fairshot (relevance order) | 0.80 | 0/5 | 969 |
| strong_fairshot_chrono (time order) | 0.96 | 4/5 | 979 |

Reference (v12): vector 0.52, episodic 1.00 ~974, validity brief 1.00 ~438.

Predictions vs results:

1. Recency ablation leaves first-city near 0 at every weight: CONFIRMED. 0/5, 0/5, 0/4. The failure is not a recency artifact.
2. First-city accuracy rises with top_k: WRONG. kFULL stayed 0/5. At full context the fact is present and still missed, so it was never a coverage problem in the way I framed it.
3. Fair-shot (full context, timestamps visible) approaches 1.00: WRONG for relevance order. It hit 0.80 and 0/5 on first-city. Visible dates were not enough.
4. RAG orders the chain only by keeping the whole history: PARTIAL. Necessary but not sufficient. Keeping everything in relevance order still failed (fair-shot 0/5). Keeping everything in chronological order recovered it (fair-shot-chrono 4/5).

The decisive variable is presentation order, the one I added after the meter run. Same retrieval, same timestamps; ordering the snippets oldest-first took first-city from 0/5 to 4/5 and overall 0.80 to 0.96. So the bottleneck is not retrieval quality, recency, context size, or even visible timestamps. It is that relevance ranking scrambles the chain, and the model does not reliably re-sort by date on its own. Chronological presentation fixes it, at keep-everything token cost (979 vs the brief's 438). The validity brief reaches the same accuracy at bounded cost because it stores the ordered lineage explicitly.

This is the spine of the Jun 16 lab note and it strengthens Claim 3: retrieval finds candidates, and order/authority, not similarity, decides the chain. Chapter 3's v12 numbers (strong_rag 0.76, first 0) still hold for the realistic top_k=5 setting; the chapter does not need rewriting, but its mechanism line ("recency cannot reach the oldest fact") can be sharpened to "relevance ranking scrambles the order, and restoring it costs the whole history." Promotion of Claim 3 to `tested` is Cynthia's call.

---

## June 16, 2026. Deterministic latest-value resolver control.

A scheduled nudge (Jun 9 and Jun 16 runs) proposed two controls written against a repo layout this project never adopted: a `bench_correction_chain.py` scoreboard with `correction_cases.json`, and standalone files under `recall_lab/experiments/`. The real harness is `eval/multiday_trial.py` and `eval/variance.py` over `scenarios/relocation_chain.json`, with controls in `recall_lab/controls/` on the `.respond` protocol. So the ideas were sound; the scaffolding in the nudge was wrong. Translated the higher-value of the two to the real structure.

Built `recall_lab/controls/deterministic.py`, the `DeterministicResolverAgent`. It answers the sharpest objection to validity-state consolidation: if the current answer is always the most recently stated value, why classify contradictions at all? Just take the latest.

How it works. On each user turn an LLM extracts value-setting (attribute, value) pairs, reading the user turn only, matching the system's source boundary. Each pair is stored with its scenario timestamp (set per day via `set_clock`, same mechanism as `strong_rag`) and turn index. At answer time, for each attribute the winner is `max(added_at, turn)` in pure Python; no model decides what is current. The resolved table, current value plus prior values per attribute, is handed to the model only to phrase the answer.

What it deliberately cannot do is the point. It has no notion of a confirmation versus a change: "yes, still Berlin" becomes another Berlin row (harmless), but an instruction like "keep the old one" has no representation, because there is no contradiction classifier to read intent. It keeps full per-attribute history so it can answer first/previous questions, but it orders that history by timestamp alone, never by a validity decision.

Wired into `multiday_trial` (`--agents deterministic`, folded into `all`) via `run_deterministic_trial`, and into `variance.py`'s lineup. Tests in `tests/test_deterministic_control.py` cover the resolve logic offline with an injected extractor: max-timestamp wins, history preserved for past questions, attributes do not collide, turn breaks ties on same-day timestamps, the JSON-fence-tolerant parser, and the API-key guard. 67 tests pass.

The Chapter 3 test it sets up. On the relocation chain, does `max(timestamp)` match Recall Lab? If yes, validity state is over-engineered for this scenario and the honest move is to say so. If it breaks, the break, an implicit correction with no clean value, a confirmation misread, an attribute the extractor splits wrong, is the argument for why a validity decision is not a timestamp sort. Not run yet; control and runner are in place.

Deferred to a separate job: the TTL-decay control (`controls/ttl.py`), the rule-based forget-on-silence baseline. That is the cheap-forgetting alternative and maps to the decay item already in protocol.md Future work.

Repo-hygiene note for the nudge routine: it keeps proposing an `experiments/` dir and a scoreboard that do not exist here, because it is reasoning from a generic template, not this repo. When acting on a nudge, translate its logic to `controls/` on `.respond` and ignore its file paths.

Pre-existing lint debt, not from this change: `controls/strong_rag.py` trips ruff E402 (a helper defined above its imports, lines 45-64). Confirmed present on main with this branch's changes stashed. Left out of this PR to keep it scoped; worth a one-line cleanup PR.

---

## June 16, 2026. v15: deterministic resolver ties Recall Lab. The relocation chain no longer separates them.

Ran the full lineup live under the pinned provider: `--agents all`, 5 runs, relocation chain, 3-call judge audit. The deterministic latest-value resolver was scored side by side with every existing control. Clean 5/5, 0 split verdicts of 175 graded answers.

Mean recall accuracy:

| Agent | mean | spread |
| --- | --- | --- |
| sliding_window_2 | 0.00 | flat |
| strong_rag_dated | 0.40 | flat |
| vector_topk_5 | 0.48 | 0.40-0.60 |
| strong_rag | 0.76 | 0.60-0.80 |
| episodic_judge | 1.00 | flat |
| deterministic | 1.00 | flat |
| recall_lab_brief_window_2 | 1.00 | flat |

The headline: deterministic `max(timestamp)` matches Recall Lab at 1.00, all five questions 5/5, including the three chain questions that break every retrieval baseline. So is the raw episodic judge. On `relocation_chain`, three very different strategies tie at the ceiling: validity-state consolidation, keep-everything-and-read, and extract-then-latest-wins.

What this means, stated honestly. The relocation chain does not justify validity-state consolidation over the two simpler baselines. A timestamp sort gets every question right here, because every change in this scenario is a clean, explicit, user-stated value update with an unambiguous attribute. That is exactly the case where "take the latest" is correct by construction. The scenario was built to expose retrieval's failure to order superseded facts, and it does that well (vector 0.48, strong_rag 0.76, both fail first-city). It was not built to separate validity reasoning from a timestamp sort, and it does not.

This is a falsification-style result for the current scenario, and it is the right thing to surface, not bury. Recall Lab still wins the thing it was designed for against the baselines everyone actually ships (retrieval). It does not yet beat the two strongest non-retrieval baselines, because the test is too easy for them.

What separates deterministic from validity state, by construction, and is therefore the next scenario:
- A confirmation that is not a change. "Yes, still Berlin" adds a Berlin row; harmless for latest-wins. But "ignore my last message, keep the old address" has no representation in a timestamp sort: the latest statement is not the current value. Validity state can read that intent; max(timestamp) cannot.
- An implicit correction with no clean value. "Actually that gift is for my brother, not my son" updates an attribute without restating it as attribute=value. The extractor may miss it; the contradiction classifier is built to catch it.
- A re-assertion of an old fact. Mentioning Lagos again late, not as a move back but in passing, lifts its timestamp and can make a stale fact look current to max(timestamp). This is the interference case from the May 20 activation work, now at the resolver layer.

Next step: build an adversarial scenario, `scenarios/correction_intent.json` or similar, carrying at least one confirmation-not-change, one implicit correction, and one stale re-assertion. Run the same lineup. The prediction: deterministic and episodic drop on those items, Recall Lab holds. If Recall Lab also drops, the validity mechanism has a real gap and the May 24 contradiction design needs revisiting. Either outcome is a Chapter 3 result. Not built yet.

Token-cost note, still relevant even though accuracy ties. Episodic injects the whole log every turn; deterministic carries a compact resolved table; the brief is bounded. On this short chain the input-token gap is small, but it widens with conversation length, which is the cost half of the Chapter 3 argument. Worth charting on a longer scenario once the adversarial one exists.

---

## June 16, 2026. Adversarial scenario built and smoke-validated; full campaign pending credits.

Built `scenarios/correction_intent.json`, the adversarial scenario the v15 entry called for. Same persona and four-day shape as the relocation chain, but three of its final-eval questions are designed to separate a validity decision from a max(timestamp) sort:

- Stale re-assertion (expect blue): color is changed green to blue, then green is fondly re-mentioned in passing without changing the preference. A timestamp sort can lift the late green mention and return it.
- Revert (expect Berlin): shipping is Berlin, switched to Munich, then the Munich change is cancelled without restating Berlin. The latest value-set is Munich, so a timestamp sort returns Munich.
- Implicit correction by negation (expect father): the milder discriminator; the value is stated so a timestamp sort should handle it.

Two controls: a confirmation-not-a-change (shellfish, explicitly reaffirmed) and a history question (original color before the change).

Smoke validation, one deterministic run before credits ran out: deterministic scored 0.4, down from 1.0 on the relocation chain. It failed exactly the two predicted cases, returning green for the stale re-assertion and Munich for the revert, and passed father, shellfish, and inverted the history answer as a knock-on of the green mistake. So the scenario discriminates as designed: the timestamp sort that tied Recall Lab on the easy chain breaks here.

The full seven-agent campaign (`--agents all`, 5 runs) did not complete: all runs failed with OpenRouter HTTP 402, out of credits. The open question stands until it runs: does Recall Lab hold on blue and Berlin where the timestamp sort breaks? If yes, this is the scenario that separates validity-state reasoning from recency. If Recall Lab also drops, the contradiction and supersede design has a real gap, most likely the sleep job promoting a passing mention (blue case) or failing to read a value-less cancellation as a revert (Berlin case). Rerun the full lineup once credits are topped up.

---

## June 17, 2026. v16 adversarial result: the revert is handled, the stale re-assertion is not. A real bug found.

Ran the full lineup on `correction_intent` under the pinned provider: `--agents all`, 5 runs, 3-call judge audit. 1 split verdict of 175 (a sliding-window control answer, immaterial).

Mean recall accuracy:

| Agent | mean | spread |
| --- | --- | --- |
| sliding_window_2 | 0.20 | flat |
| strong_rag_dated | 0.24 | 0.00-0.40 |
| deterministic | 0.40 | flat |
| recall_lab_brief_window_2 | 0.64 | 0.60-0.80 |
| vector_topk_5 | 0.72 | 0.60-0.80 |
| strong_rag | 0.80 | flat |
| episodic_judge | 0.92 | 0.80-1.00 |

This is not the predicted clean win, and the honest read is more useful than the prediction would have been.

Per question, the two discriminators split:

Revert (expect Berlin, "cancel the Munich change, leave it as before"):
- recall_lab 5/5. It read the value-less cancellation as a revert and kept Berlin current.
- deterministic 0/5, episodic 3/5, every retrieval baseline 0/5. max(timestamp) returned Munich as predicted; retrieval cannot represent a cancellation at all.
- This is the case validity state was supposed to win, and it did, cleanly, where everything else failed.

Stale re-assertion (expect blue, color changed green->blue then green re-mentioned in passing):
- recall_lab 0/5. It failed, and worse than a timestamp sort fails it. Inspecting the brief: the sleep job promoted the Jun 14 passing mention of green and the contradiction classifier labelled it a CORRECTION, so green was made active again and blue was demoted to "Past step 3". The lineage is inverted: active says green, past says blue, exactly backwards. Consistent across all five runs.
- deterministic 0/5 (returned green from the late mention), episodic 5/5, strong_rag 5/5.
- So the scenario's interference case defeats Recall Lab through its own consolidation path. This is the May 20 interference failure and the v8 self-poisoning failure resurfacing at the contradiction layer: a passing re-mention is not a correction, but the classifier read it as one.

Net: Recall Lab is the only agent that handled the revert, and one of several that failed the stale re-assertion, in its case via a wrong CORRECT classification rather than a timestamp artifact. episodic (0.92) and strong_rag (0.80) score higher overall here because they pass the stable/history questions and the re-assertion without a consolidation step that can mis-promote.

What this means for the thesis. The relocation chain showed validity state was not needed (everything tied at 1.0). This scenario shows validity state both helps and hurts: it uniquely solves the revert, and it introduces a failure the simpler agents do not have, because consolidation can promote the wrong thing. That is a sharper, more honest Chapter 3 than "the brief wins": authority handling is real and necessary for reverts, and the salience-to-correction path needs a guard against treating a passing re-mention as a correction.

The bug, precisely. In `consolidation/`, a Jun 14 user line that fondly re-mentions an old value ("green is still such a beautiful color, I always come back to it") is (a) promoted by the salience judge and (b) classified CORRECT against the current blue trace, superseding it. Fix direction: the contradiction classifier, or a guard before it, must distinguish a value-setting statement from a sentiment/reminiscence mention. A re-mention that does not assign the attribute as the current value should be UNRELATED, not CORRECT. This is a classifier-intent problem, the same shape as the user-only salience fix: the system must read whether the user is setting a value or just talking about one.

Next step: write a failing test that reproduces the green-re-mention -> wrong CORRECT classification at the `consolidation/contradiction.py` level (offline, deterministic), then fix the classifier prompt or add a value-setting guard, then rerun v16 and confirm recall_lab recovers the blue case without losing the revert. Do not widen any claim until that holds.

---

## June 17, 2026. Trace analysis of the v15+v16 campaigns (Langfuse export). Two real findings, one false alarm caught.

Imported the Langfuse traces/observations for every model call in the v15 and v16 campaigns (5797 calls, 2026-06-16..17) and analysed them. Hard numbers are exact from the observation records; qualitative findings were produced by a multi-agent pass and then re-verified by hand against the raw export before being trusted.

Cost and instrumentation (exact):
- 5797 calls, $3.35 total. The judge (anthropic/claude-sonnet-4.6, 2166 calls) is $3.08, 92% of spend. The agent (openai/gpt-4o-mini, 3631 calls) is $0.27. Eval cost, not agent volume, dominates, and it scales with campaign size. Before the 30-conversation campaign, tier the judge (cheap-first, escalate to Sonnet only on disagreement) or the bill grows fast.
- Latency healthy: mean 1.43s, p95 2.97s, one 20.2s outlier. The only 5 ERROR records are the HTTP 402 out-of-credits from the killed v16 first attempt. Nothing else hiding.
- The Langfuse auto-eval fields (faithfulness, groundedness, completeness, usefulness, technical_depth, context_precision, format_quality, quality_gate_attempt_*) are in the schema but 0 of 5797 are populated. Either the judge structured output is dropped on write or never requested. We are advertising evaluation coverage we do not have; fix before relying on it for regression gating.

Finding that refines the v16 bug (verified against raw traces):
- The blue stale-re-assertion failure is two-layer, not a classifier-only bug. I had logged it as "the contradiction classifier labelled it CORRECT". The traces show the failure starts one layer up. The salience judge scored the reminiscence line "green is still such a beautiful color, I always come back to it in my head" at 0.55 with the reason "User expresses a stable aesthetic preference for the color green", and promoted it as a preference. The contradiction classifier then received "favorite color is green" against active "blue" and returned CORRECT with reason "the new statement says the favorite color is green, contradicting the old fact that it was blue". Given that input, CORRECT is locally right. So the root cause is salience promoting a sentiment as a value-setting preference; the classifier mislabel is downstream of that. The fix from the June 17 spawned task still applies but the primary target moves upstream: salience must not promote a reminiscence as a value. A classifier-prompt guard is the secondary defence, not the fix.
- Verified true and useful: extractor attribute naming is stable ("shipping city" used identically across the Berlin and Munich turns, 3/3), so the deterministic 0.40 is not an attribute-splitting artifact. Do not spend effort there.

False alarm, caught and discarded:
- The multi-agent pass claimed the shellfish Never-Repeat rule was silently demoted to the Past section (a new safety bug). Direct check of all 300 recall-agent briefs in the export: 207 have shellfish correctly in "Things to never repeat", 0 have it demoted to Past. The 5 apparent hits were the prompt's own instructions mentioning the Past section, not the brief body. The claim came from one agent misreading section boundaries in a 30-record slice and generalising. Recorded so the claim does not resurface: shellfish Never-Repeat held across the whole campaign. This is why surprising agent claims get checked against the full data before they enter the log.

Net: no published number changes. The one writeup refinement is the blue-case framing, from "classifier mislabel" to "salience promoted a reminiscence as a preference, surfaced as a classifier CORRECT". Cost tiering and the dead auto-evaluators are the two cheap instrumentation actions before scaling up.

---

## June 17, 2026. v16 post-fix: the salience guard works. Recall Lab is the only agent at 1.00 on the adversarial scenario.

Reran the full lineup on `correction_intent` after the salience value_setting guard, the classifier reminiscence guard, and the invariant linter (commit cf71db1). `--agents all`, 5 runs, pinned provider, 3-call judge audit. 0 split verdicts of 175.

Recall Lab, per question, pre-fix -> post-fix:
- blue (stale re-assertion): 0/5 -> 5/5. Fixed.
- Berlin (revert): 5/5 -> 5/5. Held. The fix did not cost the revert.
- father (implicit correction): 5/5 -> 5/5.
- shellfish (confirmation control): 5/5 -> 5/5.
- green-as-history: 1/5 -> 5/5. Recovered as a consequence: once the passing green reminiscence stopped being promoted as the current preference, the active/past lineage stopped inverting, so the history question reads correctly too.

Mean recall accuracy, post-fix: sliding 0.20, strong_rag_dated 0.36, deterministic 0.40, vector 0.68, strong_rag 0.72, episodic 0.92, recall_lab 1.00.

This is the result the adversarial scenario was built to produce. Recall Lab is now the only agent at 1.00, and it got there by handling the two cases that defeat a timestamp sort: the revert (a cancellation that sets no value, deterministic 0/5) and the stale re-assertion (a fond mention that sets no value, deterministic 0/5). Both are the same underlying principle the lab keeps returning to: a statement only changes memory if it sets a value, and authority is about intent, not recency. The fix encoded that at the salience layer, where the v16 trace analysis showed the failure actually began.

What changed and why it is the right fix, not a scenario-specific patch:
- The guard is a general rule (value-setting vs sentiment), not a hard-coded green/blue case. It blocks any reminiscence from being filed as a current value.
- The Berlin revert held, so the guard did not over-suppress real changes.
- The deterministic and episodic baselines were unchanged by the fix (0.40 and 0.92), confirming the scenario did not get easier; Recall Lab moved.

The deterministic 0.40 and episodic 0.92 are the standing comparison. Deterministic still fails both discriminators because max(timestamp) cannot represent a value-less cancellation or tell a reminiscence from a value-set. Episodic passes the re-assertion and history but still drops the revert 2/5, because reading the whole raw log does not force a single authority decision. Only the validity brief gets both, every run.

Next: this is now a clean two-scenario story. relocation_chain shows validity state is not needed when every change is an explicit value-set (everything ties at 1.00). correction_intent shows it is needed and sufficient when changes are adversarial (only recall_lab holds at 1.00). That pair is the Chapter 3 result. README headline updated. The brief-invariant linter (memory/invariants.py) is in place but not yet wired into the sleep job as a runtime assertion; wiring it to fail loudly on a violation during consolidation is a small follow-up.

---

## August 19, 2026. Documentation sync audit. One unlogged change, and every published number re-verified.

No new experiments. This is a state-of-the-repo audit before the Recall Lab result goes out as a paper, so nothing gets published against a stale claim.

### The one real gap: the invariant linter was wired and never logged

The June 17 v16 post-fix entry ends with "the brief-invariant linter (`memory/invariants.py`) is in place but not yet wired into the sleep job as a runtime assertion; wiring it to fail loudly on a violation during consolidation is a small follow-up."

That follow-up was done the same evening, in commit `a3492f7`, and no log entry was written for it. The log has been wrong on this point for two months. What actually shipped:

- `run_sleep_job` now runs `check_invariants` after every consolidation pass.
- Default is **non-strict**: violations are reported in the returned summary dict under `invariant_violations` and warned to stdout, but do not raise. This is deliberate. A hard raise mid-pass would abort a whole variance campaign and lose the other runs' data over one bad pass.
- `strict=True` raises `BriefInvariantError` instead, for tests and CI.
- Tests cover a seeded active/past overlap being reported in non-strict mode and raising in strict mode, and a clean store reporting nothing.

So the answer to "is the linter a runtime assertion yet" is yes, in reporting mode, since June 17. Suite is at 81 passing, ruff clean.

### Every published number re-verified against `reports/`, not against this log

Each figure that is about to appear in an external writeup was checked against the campaign artifacts rather than re-read from the prose here. All of them hold:

| Claim | Source artifact | Verified |
| --- | --- | --- |
| v12 relocation lineup: sliding 0.00, vector 0.52, strong_rag 0.76, episodic 1.00, brief 1.00 | `v12_chapter3_lineup/variance_summary.md` | exact |
| v15 relocation lineup: + strong_rag_dated 0.40, vector 0.48, deterministic 1.00 | `v15_deterministic_all/variance_summary.md` | exact |
| v16 post-fix adversarial: sliding 0.20, dated 0.36, deterministic 0.40, vector 0.68, strong_rag 0.72, episodic 0.92, brief 1.00 | `v16_correction_intent_postfix/variance_summary.md` | exact |
| Judge audit, v16 post-fix: 0 splits of 175 | same | exact |

One number that existed only as ambiguous prose is now pinned. The June 17 post-fix entry said episodic "still drops the revert 2/5", which reads either as *scores* 2/5 or *fails* 2 of 5. The artifact settles it: on "What city should you ship to right now?" episodic_judge passes **3/5**. Recall Lab passes 5/5; deterministic and all four retrieval controls pass 0/5.

Also worth recording from the per-question table, because it is a coverage artifact and not a result: `sliding_window_2` scores 5/5 on the current-colour question in `correction_intent`. A two-turn window happens to span the relevant turn for that one question, which is why its mean is 0.20 rather than 0.00. It is not evidence of anything about recency windows.

### Docs brought back in sync in this pass

- `protocol.md` — did not describe `correction_intent` at all, did not carry the value-setting guard as part of the mechanism under test, and its falsification section was still purely forward-looking even though a falsification event (v15) has already happened and been reported. All three fixed.
- `README.md` — the "Current public read" block described only the relocation chain and still carried the caveat "one relocation scenario", while §Controls elsewhere in the same file correctly described the two-scenario result. Internally contradictory; the public read now states the two-scenario claim. `memory/invariants.py` added to Working now.
- `diagrams/README.md` — §5 was still "v10 results", six campaign versions behind. Replaced with the current two-scenario pair.

### Status of the public claim after this audit

Unchanged in substance. The claim is the two-scenario pair, and no figure moved. What changed is that the docs now say the same thing as the artifacts.
