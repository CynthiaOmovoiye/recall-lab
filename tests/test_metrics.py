"""Tests for the deterministic scoring helpers.

classify_failure_mode is the no-API-key fallback scorer. JudgeResult.unanimous
backs the judge audit. Both are pure.
"""

from __future__ import annotations

from recall_lab.eval.metrics import (
    FailureMode,
    JudgeResult,
    classify_failure_mode,
)


def test_classify_is_correct_when_the_expected_fact_is_present() -> None:
    assert classify_failure_mode("Your favorite color is blue.", "blue") is FailureMode.CORRECT


def test_classify_is_honest_gap_on_admitted_uncertainty() -> None:
    mode = classify_failure_mode("I don't know where you shipped before.", "Lagos")
    assert mode is FailureMode.HONEST_GAP


def test_classify_is_hallucinated_when_the_fact_is_absent_and_confident() -> None:
    assert classify_failure_mode("You shipped to Paris.", "Lagos") is FailureMode.HALLUCINATED


def test_judge_result_unanimous_only_when_every_vote_agrees() -> None:
    agreed = JudgeResult(mode=FailureMode.CORRECT, votes=[FailureMode.CORRECT] * 3)
    split = JudgeResult(
        mode=FailureMode.CORRECT,
        votes=[FailureMode.CORRECT, FailureMode.CORRECT, FailureMode.HALLUCINATED],
    )
    assert agreed.unanimous is True
    assert split.unanimous is False
