"""Mean and persona vectors are centered on the neutral Arabic control by default."""

import logging
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from machine_poi.controls import neutral_texts, texts_sha256
from machine_poi.llm_wrapper import SteeredLLM
from machine_poi.quran_embeddings import QuranEmbeddings
from machine_poi.steerer import InvalidConfigError, QuranSteerer
from machine_poi.steering_cache import save_vectors

GENERIC, QURAN, NEUTRAL = torch.eye(4)[:3]
CONTROL = set(neutral_texts("ar"))


@pytest.fixture
def steerer():
    """Activations share a large generic component; only the content part differs."""
    steerer = QuranSteerer(device="cpu")
    steerer.embedder = Mock(spec=QuranEmbeddings)
    steerer.embedder.load_quran_text.return_value = [f"verse {i}" for i in range(12)]
    steerer.llm = Mock(spec=SteeredLLM)
    steerer.llm.hidden_size, steerer.llm.num_layers = 4, 2
    steerer.llm.layer_token_norms.return_value = {0: 10.0, 1: 10.0}
    steerer.pooled_batches = []

    def pooled(texts, **kwargs):
        steerer.pooled_batches.append(list(texts))
        rows = [10 * GENERIC + (NEUTRAL if text in CONTROL else QURAN) for text in texts]
        return {layer: torch.stack(rows) for layer in range(2)}

    steerer.llm.pooled_layer_means.side_effect = pooled
    return steerer


CENTERED = torch.nn.functional.normalize(QURAN - NEUTRAL, dim=0)
RAW = torch.nn.functional.normalize(10 * GENERIC + QURAN, dim=0)


def test_quran_steering_subtracts_the_neutral_control_mean(steerer):
    vectors = steerer.prepare_quran_steering(sample_size=5)
    assert [len(batch) for batch in steerer.pooled_batches] == [5, len(CONTROL)]
    assert set(steerer.pooled_batches[1]) == CONTROL
    for layer in range(2):
        torch.testing.assert_close(vectors[layer], CENTERED)


def test_raw_mean_is_kept_behind_a_warning(steerer):
    with pytest.warns(UserWarning, match="raw_mean"):
        vectors = steerer.prepare_quran_steering(sample_size=5, recipe="raw_mean")
    assert len(steerer.pooled_batches) == 1  # no control pass
    torch.testing.assert_close(vectors[0], RAW)
    assert float(vectors[0] @ GENERIC) > 0.99  # mostly the shared component

    with pytest.raises(InvalidConfigError, match="recipe"):
        steerer.prepare_quran_steering(recipe="centred")


def test_persona_centers_every_resolution(steerer, tmp_path):
    vectors = steerer.prepare_quran_persona(cache_dir=str(tmp_path))
    assert sum(set(batch) == CONTROL for batch in steerer.pooled_batches) == 1
    torch.testing.assert_close(vectors[1], CENTERED)


def test_cache_records_the_recipe_and_rejects_older_formats(steerer, tmp_path, caplog):
    cache = tmp_path / "mean.npz"
    steerer.prepare_quran_steering(sample_size=5, cache_path=cache)
    metadata = steerer._cache_metadata(
        "mean", recipe="centered", neutral_sha256=texts_sha256(neutral_texts("ar")),
        chunk_by="verse", sample_size=5, seed=42,
    )
    assert metadata["format"] == 3

    # A matching cache is reused without another forward pass.
    steerer.pooled_batches.clear()
    steerer.prepare_quran_steering(sample_size=5, cache_path=cache)
    assert steerer.pooled_batches == []

    # A raw-mean request does not reuse centered vectors.
    with pytest.warns(UserWarning), caplog.at_level(logging.WARNING):
        steerer.prepare_quran_steering(sample_size=5, cache_path=cache, recipe="raw_mean")
    assert "different parameters" in caplog.text
    assert len(steerer.pooled_batches) == 1

    # A format-2 cache (uncentered vectors) is rejected with a reason.
    save_vectors(cache, {0: np.ones(4, dtype=np.float32)}, {**metadata, "format": 2})
    caplog.clear()
    steerer.pooled_batches.clear()
    with caplog.at_level(logging.WARNING):
        vectors = steerer.prepare_quran_steering(sample_size=5, cache_path=cache)
    assert "predates format 3" in caplog.text
    assert len(steerer.pooled_batches) == 2
    torch.testing.assert_close(vectors[0], CENTERED)


def test_english_control_is_available_for_comparison(steerer):
    english = set(neutral_texts("en"))
    steerer.prepare_quran_steering(sample_size=5, control="en")
    assert set(steerer.pooled_batches[1]) == english
    with pytest.raises(ValueError, match="fr"):
        steerer.prepare_quran_steering(sample_size=5, control="fr")
