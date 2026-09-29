"""Tests for src/themes.py and the bridge paths that use it."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, create_autospec

import numpy as np
import pytest

from src.quran_embeddings import QuranEmbeddings
from src.themes import (
    DOMAIN_BRIDGE_MAP,
    QURANIC_THEMES,
    matching_keywords,
    theme_index,
)

TARGET_THEME = 3


def fake_embedder():
    """Embedder with the real interface: unknown methods raise AttributeError."""
    identity = np.eye(len(QURANIC_THEMES), dtype=np.float32)

    def create_embeddings(texts, batch_size=32, normalize=True, show_progress=True):
        if list(texts) == QURANIC_THEMES:
            return identity
        return identity[[TARGET_THEME] * len(texts)]

    embedder = create_autospec(QuranEmbeddings, instance=True)
    embedder.create_embeddings.side_effect = create_embeddings
    return embedder


def theme_calls(embedder):
    return [
        call
        for call in embedder.create_embeddings.call_args_list
        if list(call.args[0]) == QURANIC_THEMES
    ]


@pytest.mark.parametrize(
    "text",
    [
        "How do I handle terror attacks news?",
        "Describe the latest contest results",
        "The steam engine sometimes fails",
        "Please decode this message",
        "A protest about wildlife",
    ],
)
def test_keywords_do_not_match_inside_other_words(text):
    assert matching_keywords(text, DOMAIN_BRIDGE_MAP) == []


@pytest.mark.parametrize(
    "text, expected",
    [
        ("My tests keep failing while debugging", ["debug", "test"]),
        ("Our TEAMS argued about coding errors", ["error", "code", "team", "argue"]),
        ("Families under stress", ["stress", "family"]),
    ],
)
def test_keywords_match_whole_words_and_inflections(text, expected):
    assert matching_keywords(text, DOMAIN_BRIDGE_MAP) == expected


def test_theme_index_is_normalized_and_computed_once_per_embedder():
    embedder = fake_embedder()
    first = theme_index(embedder)
    second = theme_index(embedder)
    assert first is second
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-5)
    assert len(theme_calls(embedder)) == 1


def test_steerer_embedding_fallback_uses_real_embedder_interface():
    from src.steerer import QuranSteerer

    steerer = QuranSteerer()
    steerer.embedder = fake_embedder()
    for _ in range(2):
        bridges = steerer.generate_domain_bridges("Explain photosynthesis", max_bridges=3)
        assert bridges == [QURANIC_THEMES[TARGET_THEME]]
    assert len(theme_calls(steerer.embedder)) == 1


def test_graph_bridge_embedding_fallback_caches_theme_index():
    from src.graph_bridge import GraphBridgeGenerator

    lightrag = MagicMock()
    lightrag.query = AsyncMock(return_value={"answer": None})
    generator = GraphBridgeGenerator(lightrag, embedder=fake_embedder())
    for _ in range(2):
        result = asyncio.run(generator.generate_bridges("Explain photosynthesis"))
        assert result.bridges == [QURANIC_THEMES[TARGET_THEME]]
    assert len(theme_calls(generator.embedder)) == 1


def test_graph_term_extraction_uses_word_boundaries():
    from src.graph_bridge import GraphBridgeGenerator

    generator = GraphBridgeGenerator(MagicMock())
    assert generator._extract_query_terms("Terror in the meetings") == ["meeting"]
