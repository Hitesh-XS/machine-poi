"""Graph bridges come from real graph neighbors, then documented fallbacks."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from machine_poi.graph_bridge import BridgeResult, GraphBridgeGenerator
from machine_poi.hybrid_knowledge_base import HybridQuranKnowledgeBase


def edge(source, target, keywords, weight=1.0):
    return SimpleNamespace(
        source=source, target=target, type=None,
        properties={"keywords": keywords, "weight": weight},
    )


GRAPHS = {
    "Tawakkul": SimpleNamespace(nodes=[], edges=[
        edge("Tawakkul", "Trust In Allah", "is aspect of", 1.0),
        edge("Tawakkul", "Effort", "requires", 3.0),
        edge("Prayer", "Tawakkul", "mentioned with", 1.5),
        edge("Effort", "Reward", "leads to", 9.0),  # not a direct neighbor
    ]),
    "Gratitude": SimpleNamespace(nodes=[], edges=[edge("Gratitude", "Increase", "leads to", 1.0)]),
}


def generator(labels=("Tawakkul", "Gratitude", "Effort", "Prayer")):
    lightrag = MagicMock()
    lightrag.get_graph_labels = AsyncMock(return_value=list(labels))
    lightrag.get_entity_neighbors = AsyncMock(
        side_effect=lambda entity_name, **kw: GRAPHS[entity_name]
    )
    lightrag.query = AsyncMock()
    return GraphBridgeGenerator(lightrag)


def bridges(gen, query):
    return asyncio.run(gen.generate_bridges(query, max_bridges=3))


def test_mapped_terms_bridge_to_ranked_graph_neighbors():
    gen = generator()
    result = bridges(gen, "I feel stress before my deadline")  # stress -> tawakkul
    # Thematic relations score double: Effort 6.0, Prayer 1.5, Trust In Allah 2.0
    assert result.bridges == ["Effort", "Trust In Allah", "Prayer"]
    assert result.confidence_scores["Effort"] == 1.0
    assert result.entities_found == ["Tawakkul"]
    assert ("Tawakkul", "requires", "Effort") in result.relationships_traversed
    assert "Reward" not in result.bridges
    gen.lightrag.query.assert_not_called()  # no LLM call to guess entities


def test_graph_labels_named_in_the_query_are_seeds():
    result = bridges(generator(), "What does the Quran say about gratitude?")
    assert result.bridges == ["Increase"] and result.entities_found == ["Gratitude"]


def test_unavailable_graph_falls_back_to_unverified_seed_concepts():
    gen = generator()
    gen.lightrag.get_graph_labels = AsyncMock(side_effect=RuntimeError("graph not built"))
    result = bridges(gen, "I feel stress")
    assert result.bridges == ["sabr", "tawakkul", "peace"]
    assert set(result.confidence_scores.values()) == {0.5}
    assert result.entities_found == [] and result.relationships_traversed == []


def test_hybrid_query_reports_traversed_graph_entities_and_relationships():
    kb = HybridQuranKnowledgeBase()
    kb._initialized = True
    kb._vector_kb = MagicMock()
    kb._vector_kb.query_with_bridges.return_value = {"verse": [], "passage": [], "surah": []}
    kb._bridge_generator = MagicMock()
    kb._bridge_generator.generate_bridges = AsyncMock(
        return_value=BridgeResult(
            bridges=["Effort"],
            entities_found=["Tawakkul"],
            relationships_traversed=[("Tawakkul", "requires", "Effort")],
            confidence_scores={"Effort": 1.0},
        )
    )
    result = asyncio.run(kb.query("stress", mode="vector"))
    assert result.graph_entities == ["Tawakkul"]
    assert result.graph_relationships == [("Tawakkul", "requires", "Effort")]
    assert not hasattr(result, "fusion_strategy")
