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
