"""Language-matched neutral controls for contrastive steering."""

import json
import re
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from machine_poi.controls import neutral_texts, texts_sha256, unique_texts
from machine_poi.llm_wrapper import SteeredLLM
from machine_poi.quran_embeddings import QuranEmbeddings

CORPUS = Path(__file__).resolve().parents[1] / "al-quran.txt"
RELIGIOUS_WORDS = {
    "الله", "لله", "القرآن", "قرآن", "الصلاة", "صلاة", "المسجد", "مسجد",
    "النبي", "نبي", "الجنة", "آية", "سورة", "رمضان", "الدين", "الإيمان",
}


def words(text):
    return re.findall(r"[ء-ي]+", text)


def test_arabic_control_is_distinct_arabic_prose_without_religious_vocabulary():
    texts = neutral_texts("ar")
    assert len(texts) >= 100
    assert len(set(texts)) == len(texts)
    for text in texts:
        assert not re.search(r"[A-Za-z]", text), text
        assert len(words(text)) >= 4, text
        assert not set(words(text)) & RELIGIOUS_WORDS, text
    assert neutral_texts() == texts  # Arabic is the default


def test_arabic_control_shares_no_four_word_run_with_the_corpus():
    def runs(text):
        tokens = words(text)
        return {tuple(tokens[i:i + 4]) for i in range(len(tokens) - 3)}

    corpus = CORPUS.read_text(encoding="utf-8").splitlines()
    corpus_runs = set().union(*(runs(line) for line in corpus))
    assert not [text for text in neutral_texts("ar") if runs(text) & corpus_runs]


def test_english_control_and_unknown_languages():
    assert len(neutral_texts("en")) == 10
    with pytest.raises(ValueError, match="fr"):
        neutral_texts("fr")


def test_unique_texts_and_hash():
    assert unique_texts([" a ", "b", "a", "", "  "]) == ["a", "b"]
    assert texts_sha256(["a", "b"]) != texts_sha256(["b", "a"])
    assert len(texts_sha256([])) == 64


@pytest.fixture
def contrastive_steerer(sample_quran_path):
    from machine_poi.steerer import ContrastiveQuranSteerer

    steerer = ContrastiveQuranSteerer(quran_path=sample_quran_path)
    steerer.embedder = Mock(spec=QuranEmbeddings)
    steerer.embedder.load_quran_text.return_value = [f"verse {i}" for i in range(80)]
    steerer.llm = Mock(spec=SteeredLLM)
    steerer.llm.hidden_size, steerer.llm.num_layers = 8, 2
    steerer.pooled_batches = []

    def pooled(texts, **kwargs):
        steerer.pooled_batches.append(list(texts))
        return {layer: torch.randn(len(texts), 8) for layer in range(2)}

    steerer.llm.pooled_layer_means.side_effect = pooled
    steerer.llm.layer_token_norms.return_value = {0: 10.0, 1: 10.0}
    steerer.device = "cpu"
    return steerer


def test_quran_contrast_uses_distinct_arabic_controls_by_default(contrastive_steerer):
    contrastive_steerer.prepare_quran_contrastive(quran_sample_size=20, neutral_sample_size=30)
    positives, negatives = contrastive_steerer.pooled_batches
    assert len(positives) == 20
    assert len(negatives) == len(set(negatives)) == 30
    assert set(negatives) <= set(neutral_texts("ar"))
    assert all(type(text) is str for text in positives + negatives)


def test_contrastive_negatives_are_deduplicated_and_hashed(contrastive_steerer, tmp_path):
    cache = tmp_path / "contrastive.npz"
    contrastive_steerer.prepare_contrastive_steering(
        ["verse 1", "verse 2"], ["neutral", "neutral", " other ", ""], cache_path=cache
    )
    assert contrastive_steerer.pooled_batches[1] == ["neutral", "other"]
    with np.load(cache) as data:
        metadata = json.loads(str(data["metadata"].item()))
    assert metadata["parameters"] == {
        "positive_sha256": texts_sha256(["verse 1", "verse 2"]),
        "negative_sha256": texts_sha256(["neutral", "other"]),
    }

    with pytest.raises(ValueError, match="negative_texts"):
        contrastive_steerer.prepare_contrastive_steering(["verse 1"], ["", "  "])
