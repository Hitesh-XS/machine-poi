"""Metrics for the steering evaluation harness (``experiments/steering_eval.py``).

Text metrics (script share, distinct-n, degeneration) and the statistics are
deterministic and need no model. The log-probability metrics run a
``SteeredLLM`` under whatever hooks are enabled. Confidence intervals are
percentile bootstraps over prompts or items from :mod:`transport_stats`.

Script share is not language identification: it counts Arabic-script and
Latin letters, which separates Arabic from English output but not, for
example, Arabic from Persian. That is the distinction the Quran/neutral
contrasts can confound.
"""

import math
import re
import unicodedata
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .transport_stats import bootstrap_ci, paired_test

ARABIC_LETTER = re.compile(
    "[ء-يٱ-ۓۺ-ۿݐ-ݿࢠ-ࣿ"
    "ﭐ-ﷻﹰ-ﻼ]"
)
LATIN_LETTER = re.compile("[A-Za-zÀ-ɏ]")
ARABIC_MARKS = re.compile("[ً-ٰٟـ]")  # diacritics and tatweel
WORD = re.compile(r"[^\W_]+")
REPEATED_UNIT = re.compile(r"(.{1,4}?)\1{7,}", re.DOTALL)


def script_counts(text: str) -> Dict[str, int]:
    """Number of Arabic-script and Latin letters in ``text``."""
    return {
        "arabic": len(ARABIC_LETTER.findall(text)),
        "latin": len(LATIN_LETTER.findall(text)),
    }


def arabic_share(text: str) -> float:
    """Arabic-script letters as a fraction of Arabic and Latin letters; NaN if none."""
    counts = script_counts(text)
    total = counts["arabic"] + counts["latin"]
    return counts["arabic"] / total if total else float("nan")


def dominant_script(text: str, threshold: float = 0.5) -> str:
    """"arabic" when at least ``threshold`` of letters are Arabic script, else "latin"."""
    share = arabic_share(text)
    if math.isnan(share):
        return "none"
    return "arabic" if share >= threshold else "latin"


def words(text: str) -> List[str]:
    """Lower-cased word tokens; Arabic diacritics are dropped so words stay whole."""
    return WORD.findall(ARABIC_MARKS.sub("", text).lower())


def distinct_n(tokens: Sequence[str], n: int) -> float:
    """Unique n-grams over all n-grams; NaN when there are fewer than ``n`` tokens."""
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(grams)) / len(grams) if grams else float("nan")


def character_loop(text: str) -> bool:
    """A 1-4 character unit with a letter or mark, repeated 8+ times in a row.

    Catches collapse inside a single "word" ("ororororor", "θθθθθθθθ", a run
    of Arabic combining marks), which word n-grams miss; runs of punctuation
    or digits such as markdown rules do not count.
    """
    return any(
        any(unicodedata.category(char)[0] in "LM" for char in match.group(1))
        for match in REPEATED_UNIT.finditer(text)
    )


def degenerate(text: str, min_words: int = 5, min_distinct_2: float = 0.5) -> bool:
    """Collapse detector: too few words, mostly repeated bigrams, or a character loop.

    With greedy decoding of 60 or more tokens, fluent output stays well above
    the word thresholds; loops such as "the the the" or a repeated phrase fall
    below the distinct-2 threshold. It is a lower bound: gibberish without a
    loop passes, which the NLL metric reflects instead.
    """
    tokens = words(text)
    if len(tokens) < min_words or character_loop(text):
        return True
    return distinct_n(tokens, 2) < min_distinct_2


def summarize(values: Iterable[float], n_boot: int = 2000, seed: int = 0) -> Dict:
    """Mean with a 95% percentile-bootstrap interval, ignoring NaN values."""
    finite = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not finite:
        nan = float("nan")
        return {"n": 0, "mean": nan, "ci_low": nan, "ci_high": nan}
    mean, low, high = bootstrap_ci(finite, n_boot=n_boot, seed=seed)
    return {"n": len(finite), "mean": mean, "ci_low": low, "ci_high": high}


def paired_difference(
    condition: Sequence[float],
    baseline: Sequence[float],
    n_boot: int = 2000,
    n_perm: int = 5000,
    seed: int = 0,
) -> Dict:
    """Mean condition-minus-baseline difference over pairs where both are finite."""
    if len(condition) != len(baseline):
        raise ValueError("Paired metrics need one value per prompt in each condition")
    diffs = [
        float(c) - float(b)
        for c, b in zip(condition, baseline)
        if c is not None and b is not None and not math.isnan(float(c)) and not math.isnan(float(b))
    ]
    if not diffs:
        nan = float("nan")
        return {"n": 0, "mean_diff": nan, "ci_low": nan, "ci_high": nan, "p_value": nan}
    return paired_test(diffs, n_boot=n_boot, n_perm=n_perm, seed=seed).to_dict()


def cohens_kappa(
    rater_a: Sequence, rater_b: Sequence, weights: Optional[str] = None
) -> float:
    """Cohen's kappa between two raters; ``weights="quadratic"`` for ordinal scores."""
    if len(rater_a) != len(rater_b) or not rater_a:
        raise ValueError("Kappa needs two equally long, non-empty rating lists")
    categories = sorted(set(rater_a) | set(rater_b))
    index = {category: i for i, category in enumerate(categories)}
    k = len(categories)
    if k == 1:
        return 1.0
    observed = np.zeros((k, k))
    for a, b in zip(rater_a, rater_b):
        observed[index[a], index[b]] += 1
    observed /= observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    i, j = np.indices((k, k))
    if weights is None:
        disagreement = (i != j).astype(float)
    elif weights == "quadratic":
        disagreement = ((i - j) / (k - 1)) ** 2
    else:
        raise ValueError("weights must be None or 'quadratic'")
    expected_disagreement = float((disagreement * expected).sum())
    if expected_disagreement == 0:
        return 1.0
    return 1.0 - float((disagreement * observed).sum()) / expected_disagreement


def centroid_contrast(
    embeddings: np.ndarray, positive: np.ndarray, negative: np.ndarray
) -> np.ndarray:
    """Cosine to the positive centroid minus cosine to the negative one, per row."""
    def unit(x):
        return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)

    rows = unit(np.asarray(embeddings, dtype=np.float64))
    return rows @ unit(np.asarray(positive, dtype=np.float64)) - rows @ unit(
        np.asarray(negative, dtype=np.float64)
    )


def mean_token_nll(llm, context: str, continuation: str, add_special_tokens: bool) -> float:
    """Mean negative log-likelihood per continuation token; NaN for an empty one."""
    total, count = llm.continuation_logprob(
        context, continuation, add_special_tokens=add_special_tokens
    )
    return -total / count if count else float("nan")


def multiple_choice_correct(llm, question: str, choices: Sequence[str], answer: int) -> int:
    """1 if the length-normalized log-likelihood picks the right choice, else 0.

    Zero-shot "Question: ...\\nAnswer: <choice>" scoring under the current
    hooks, normalized by the choice's character length (lm-eval "acc_norm").
    """
    context = f"Question: {question}\nAnswer:"
    scores = []
    for choice in choices:
        total, _ = llm.continuation_logprob(context, f" {choice}", add_special_tokens=True)
        scores.append(total / max(len(choice), 1))
    return int(int(np.argmax(scores)) == answer)
