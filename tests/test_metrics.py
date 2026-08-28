"""Tests for the deterministic scoring helpers.

classify_failure_mode is the no-API-key fallback scorer. JudgeResult.unanimous
backs the judge audit. score_abstain_answer is the abstain axis: it scores a
question whose correct outcome is a refusal, with no judge call. All pure.
"""

from __future__ import annotations

from recall_lab.eval.metrics import (
    FailureMode,
    JudgeResult,
    classify_failure_mode,
    expects_abstain,
    looks_like_abstain,
    score_abstain_answer,
)
from recall_lab.eval.multiday_trial import score_answer


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


# --- Abstain axis -----------------------------------------------------------
#
# Every scenario question before scenarios/refusal_cases.json expected a
# positive string, so the lab never tested whether an agent knows when to hold.
# These pin the branch that scores a refusal as the correct outcome.


def test_expects_abstain_reads_the_literal_marker() -> None:
    assert expects_abstain("HONEST_GAP") is True
    assert expects_abstain("  honest_gap  ") is True
    assert expects_abstain("Berlin") is False
    assert expects_abstain(None) is False


def test_abstain_scoring_credits_a_refusal() -> None:
    result = score_abstain_answer("I don't have a billing address for you.")
    assert result.mode is FailureMode.HONEST_GAP


def test_abstain_scoring_punishes_a_confident_answer() -> None:
    result = score_abstain_answer("Your billing address is 12 Marina Road, Lagos.")
    assert result.mode is FailureMode.HALLUCINATED


def test_looks_like_abstain_covers_all_three_refusal_reasons() -> None:
    # conflict: two equally valid values, nothing says which is current
    assert looks_like_abstain("I cannot tell which of the two is your main line.")
    # absence: the fact was never stated
    assert looks_like_abstain("You never told me the dog's name.")
    # revoked: the user withdrew the fact
    assert looks_like_abstain("You asked me to forget it, so I no longer have it.")


def test_score_answer_credits_a_refusal_and_spends_no_judge_call(monkeypatch) -> None:
    """An abstain question must not reach the LLM judge.

    The judge is 92% of a campaign's cost, so a refusal question has to be
    free. This also guards against the branch silently falling through.
    """
    calls: list[tuple] = []
    monkeypatch.setattr(
        "recall_lab.eval.multiday_trial.judge_answer",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    correct, mode, votes = score_answer(
        "What is my billing address?",
        "You asked me to forget it, so I do not have it.",
        "HONEST_GAP",
        judge_samples=3,
    )

    assert correct is True
    assert mode is FailureMode.HONEST_GAP
    assert votes == ["honest_gap"]
    assert calls == []


def test_score_answer_marks_a_confident_answer_to_a_refusal_question_hallucinated(
    monkeypatch,
) -> None:
    calls: list[tuple] = []
    monkeypatch.setattr(
        "recall_lab.eval.multiday_trial.judge_answer",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    correct, mode, votes = score_answer(
        "What is my billing address?",
        "Your billing address is 12 Marina Road, Lagos.",
        "HONEST_GAP",
        judge_samples=3,
    )

    assert correct is False
    assert mode is FailureMode.HALLUCINATED
    assert votes == ["hallucinated"]
    assert calls == []


def test_score_answer_still_routes_a_normal_question_to_the_judge(monkeypatch) -> None:
    """A fact question keeps its existing path, untouched."""
    seen: dict = {}

    def fake_judge(question, response, ground_truth, samples=1):
        seen["samples"] = samples
        return JudgeResult(mode=FailureMode.CORRECT, votes=[FailureMode.CORRECT] * samples)

    monkeypatch.setattr("recall_lab.eval.multiday_trial.judge_answer", fake_judge)

    correct, mode, votes = score_answer(
        "What city should you ship to right now?", "Berlin.", "Berlin", judge_samples=3
    )

    assert correct is True
    assert mode is FailureMode.CORRECT
    assert votes == ["correct"] * 3
    assert seen["samples"] == 3
