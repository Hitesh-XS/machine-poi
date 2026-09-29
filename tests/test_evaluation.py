"""Evaluation metrics: text statistics, intervals, agreement and log-likelihoods."""

import math

import numpy as np
import pytest
import torch

from machine_poi.evaluation import (
    arabic_share,
    centroid_contrast,
    cohens_kappa,
    degenerate,
    distinct_n,
    dominant_script,
    mean_token_nll,
    multiple_choice_correct,
    paired_difference,
    summarize,
    words,
)


def test_script_share_separates_arabic_from_latin_output():
    assert arabic_share("الصبر مفتاح الفرج") == 1.0
    assert arabic_share("Patience is the key") == 0.0
    assert arabic_share("Patience الصبر") == pytest.approx(5 / 13)
    assert math.isnan(arabic_share("123 ... !"))
    assert dominant_script("الصَّبْرُ مفتاح the") == "arabic"
    assert dominant_script("mostly English with كلمة") == "latin"
    assert dominant_script("42") == "none"


def test_words_keep_diacritized_arabic_whole():
    assert words("الصَّبْرُ، مفتاحُ الفرج!") == ["الصبر", "مفتاح", "الفرج"]
    assert words("Don't STOP_now") == ["don", "t", "stop", "now"]


def test_distinct_n_and_collapse_detection():
    assert distinct_n(["a", "b", "a", "b"], 2) == pytest.approx(2 / 3)
    assert math.isnan(distinct_n(["a"], 2))
    assert degenerate("the the the the the the the the")
    assert degenerate("Too short.")
    assert not degenerate("Water boils at a lower temperature on high mountains.")
    assert degenerate("I am here. " * 10)


def test_summaries_and_paired_differences():
    summary = summarize([1.0, 2.0, 3.0, float("nan")], n_boot=500)
    assert summary["n"] == 3 and summary["mean"] == 2.0
    assert summary["ci_low"] <= 2.0 <= summary["ci_high"]
    assert summarize([float("nan")])["n"] == 0

    difference = paired_difference([2.0, 3.0, float("nan")], [1.0, 1.0, 1.0], n_boot=500)
    assert difference["n"] == 2 and difference["mean_diff"] == 1.5
    with pytest.raises(ValueError):
        paired_difference([1.0], [1.0, 2.0])


def test_cohens_kappa_matches_hand_computation():
    a = [0, 1, 2, 2, 1, 0]
    assert cohens_kappa(a, a) == 1.0
    # Observed agreement 4/6; expected from marginals (2,2,2)x(2,1,3) = 1/3.
    b = [0, 1, 2, 2, 2, 1]
    assert cohens_kappa(a, b) == pytest.approx((4 / 6 - 1 / 3) / (1 - 1 / 3))
    assert cohens_kappa(a, b, weights="quadratic") > cohens_kappa(a, b)
    with pytest.raises(ValueError):
        cohens_kappa([0], [0, 1])


def test_centroid_contrast_prefers_the_positive_side():
    rows = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    scores = centroid_contrast(rows, np.array([2.0, 0.0]), np.array([0.0, 3.0]))
    np.testing.assert_allclose(scores, [1.0, -1.0, 0.0], atol=1e-12)


def test_continuation_logprob_matches_a_manual_forward_pass(tiny_llm):
    total, count = tiny_llm.continuation_logprob("the quran", " mercy patience")
    ids = tiny_llm.tokenizer("the quran mercy patience")["input_ids"]
    with torch.no_grad():
        logits = tiny_llm.model(input_ids=torch.tensor([ids])).logits[0]
    log_probs = torch.log_softmax(logits, dim=-1)
    expected = log_probs[2, ids[3]] + log_probs[3, ids[4]]
    assert count == 2 and total == pytest.approx(float(expected), rel=1e-5)
    assert tiny_llm.continuation_logprob("the", "") == (0.0, 0)
    assert mean_token_nll(tiny_llm, "the quran", " mercy patience", True) == pytest.approx(
        -total / 2
    )


def test_log_likelihoods_follow_the_active_hooks(tiny_llm):
    plain = tiny_llm.continuation_logprob("the quran", " mercy")[0]
    tiny_llm.register_steering_hook(1, torch.ones(16), coefficient=5.0)
    try:
        steered = tiny_llm.continuation_logprob("the quran", " mercy")[0]
        with tiny_llm.steering_disabled():
            unsteered = tiny_llm.continuation_logprob("the quran", " mercy")[0]
    finally:
        tiny_llm.clear_steering()
    assert steered != pytest.approx(plain)
    assert unsteered == pytest.approx(plain)


def test_multiple_choice_scores_by_length_normalized_likelihood(tiny_llm, monkeypatch):
    scores = {" water": -1.0, " justice path": -4.0, " light": -0.2}
    monkeypatch.setattr(
        tiny_llm, "continuation_logprob", lambda context, choice, **kw: (scores[choice], 1)
    )
    choices = ["water", "justice path", "light"]
    # Per character: -0.2, -0.33, -0.04 -> "light"
    assert multiple_choice_correct(tiny_llm, "q", choices, answer=2) == 1
    assert multiple_choice_correct(tiny_llm, "q", choices, answer=0) == 0
