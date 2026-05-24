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

