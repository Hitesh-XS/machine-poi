"""
Graph-Based Domain Bridge Generator

Uses LightRAG knowledge graph to dynamically map user queries
to relevant Quranic themes through entity-relationship traversal.
"""

import logging
import re
from typing import List, Dict, Optional, Set, Tuple, TYPE_CHECKING
from dataclasses import dataclass
import numpy as np

from .async_utils import run_sync
from .themes import QURANIC_THEMES, embed_query, matching_keywords, theme_index

if TYPE_CHECKING:
    from .lightrag_adapter import QuranLightRAG
    from .quran_embeddings import QuranEmbeddings

logger = logging.getLogger("machine_poi.graph_bridge")


@dataclass
class BridgeResult:
    """Result from graph-based bridge generation."""
    bridges: List[str]
    entities_found: List[str]
    relationships_traversed: List[Tuple[str, str, str]]
    confidence_scores: Dict[str, float]


class GraphBridgeGenerator:
    """
    Generates domain bridges using knowledge graph traversal.

    Three-tier approach:
    1. Direct entity matching (fast path)
    2. Graph neighbor traversal (relationship-based)
    3. Embedding similarity fallback (when graph fails)
    """

    # Pre-defined mappings from common terms to graph entity names
    # These help map user vocabulary to Quranic entity names
    TERM_TO_ENTITY: Dict[str, List[str]] = {
        # Technical terms → Quranic entities
        "debug": ["patience", "careful_examination", "wisdom"],
        "error": ["repentance", "forgiveness", "learning"],
        "optimize": ["ihsan", "excellence", "perfection"],
        "team": ["ummah", "brotherhood", "unity"],
        "leader": ["prophet", "responsibility", "justice"],
        "stress": ["sabr", "tawakkul", "peace"],
        "failure": ["perseverance", "hope", "resilience"],
        "success": ["gratitude", "humility", "shukr"],
        # Work/career terms
        "deadline": ["time_management", "responsibility", "trust"],
        "meeting": ["consultation", "shura", "wisdom"],
        "conflict": ["reconciliation", "patience", "justice"],
        "promotion": ["gratitude", "humility", "continued_effort"],
        # Emotional terms
        "anxiety": ["tawakkul", "dhikr", "peace"],
        "fear": ["courage", "trust", "hope"],
        "anger": ["patience", "forgiveness", "self_control"],
        "sadness": ["hope", "patience", "trust_in_allah"],
    }

    # Relationship types that indicate thematic relevance
    BRIDGE_RELATIONSHIPS: Set[str] = {
        "exemplifies",
        "teaches",
        "leads_to",
        "requires",
        "contrasts_with",
        "manifests_as",
        "is_aspect_of",
        "practiced_by",
    }

    def __init__(
        self,
        quran_lightrag: "QuranLightRAG",
        embedder: Optional["QuranEmbeddings"] = None,
        max_traversal_depth: int = 2,
        min_confidence: float = 0.3,
    ):
        """
        Initialize GraphBridgeGenerator.

        Args:
            quran_lightrag: QuranLightRAG instance for graph queries
            embedder: Optional embedder for similarity fallback
            max_traversal_depth: Maximum graph traversal depth
            min_confidence: Minimum confidence threshold for bridges
        """
        self.lightrag = quran_lightrag
        self.embedder = embedder
        self.max_depth = max_traversal_depth
        self.min_confidence = min_confidence

        # Cache for graph labels (entity names)
        self._entity_cache: Optional[List[str]] = None

    async def _get_entity_names(self) -> List[str]:
        """Get all entity names from the knowledge graph."""
        if self._entity_cache is None:
            self._entity_cache = await self.lightrag.get_graph_labels()
        return self._entity_cache

    def _extract_query_terms(self, query: str) -> List[str]:
        """Extract relevant terms from user query."""
        return matching_keywords(query, self.TERM_TO_ENTITY)

    @staticmethod
    def _normalize(name: str) -> str:
        return " ".join(name.replace("_", " ").lower().split())

    async def _seed_labels(self, query: str, terms: List[str]) -> List[str]:
        """Graph labels for the mapped seed concepts and for labels named in the query."""
        try:
            labels = await self._get_entity_names()
        except Exception as e:
            logger.warning(f"Graph labels unavailable: {e}")
            return []
        by_name = {self._normalize(label): label for label in labels}
        seeds = [
            by_name[self._normalize(entity)]
            for term in terms
            for entity in self.TERM_TO_ENTITY[term]
            if self._normalize(entity) in by_name
        ]
        query_text = self._normalize(query)
        seeds += [
            label
            for name, label in by_name.items()
            if re.search(rf"\b{re.escape(name)}\b", query_text)
        ]
        return list(dict.fromkeys(seeds))

    def _neighbors(self, seed: str, graph) -> List[Tuple[str, str, float]]:
        """Direct neighbors of seed as (neighbor, relation, score).

        Edges whose keywords name a BRIDGE_RELATIONSHIPS type score double.
        """
        found = []
        for edge in getattr(graph, "edges", None) or []:
            if seed not in (edge.source, edge.target) or edge.source == edge.target:
                continue
            neighbor = edge.target if edge.source == seed else edge.source
            properties = edge.properties or {}
            relation = str(properties.get("keywords") or edge.type or "related")
            text = self._normalize(relation)
            thematic = any(self._normalize(kind) in text for kind in self.BRIDGE_RELATIONSHIPS)
            try:
                weight = float(properties.get("weight", 1.0))
            except (TypeError, ValueError):
                weight = 1.0
            found.append((neighbor, relation, weight * (2.0 if thematic else 1.0)))
        return found

    async def generate_bridges(
        self,
        query: str,
        max_bridges: int = 5,
        use_graph: bool = True,
        use_embedding_fallback: bool = True,
    ) -> BridgeResult:
        """
        Generate domain bridges for a query using the knowledge graph.

        Algorithm:
        1. Map query terms (TERM_TO_ENTITY) and graph labels named in the
           query to entities in the graph
        2. Collect the direct neighbors of up to three seed entities, ranked by
           edge weight, doubled for thematic relation types
        3. Otherwise fall back to embedding similarity with QURANIC_THEMES,
           then to the mapped seed concepts themselves

        Confidence is the neighbor's score relative to the best neighbor for
        graph bridges, cosine similarity for embedding bridges, and 0.5 for
        unverified seed concepts.

        Args:
            query: User query string
            max_bridges: Maximum number of bridges to return
            use_graph: Whether to use graph traversal
            use_embedding_fallback: Whether to fall back to embeddings

        Returns:
            BridgeResult with bridges and metadata
        """
        bridges: List[str] = []
        entities_found: List[str] = []
        relationships: List[Tuple[str, str, str]] = []
        confidence: Dict[str, float] = {}

        terms = self._extract_query_terms(query)
        seed_entities = list(
            dict.fromkeys(entity for term in terms for entity in self.TERM_TO_ENTITY[term])
        )

        if use_graph:
            scores: Dict[str, float] = {}
            for seed in (await self._seed_labels(query, terms))[:3]:
                try:
                    graph = await self.lightrag.get_entity_neighbors(
                        entity_name=seed,
                        max_depth=self.max_depth,
                        max_nodes=10,
                    )
                except Exception as e:
                    logger.debug(f"No graph data for entity '{seed}': {e}")
                    continue
                entities_found.append(seed)
                for neighbor, relation, score in self._neighbors(seed, graph):
                    relationships.append((seed, relation, neighbor))
                    scores[neighbor] = max(score, scores.get(neighbor, 0.0))
            ranked = sorted(
                (name for name in scores if name not in entities_found),
                key=lambda name: -scores[name],
            )[:max_bridges]
            if ranked:
                best = scores[ranked[0]]
                bridges = ranked
                confidence = {name: scores[name] / best for name in ranked}

        # Fallback to embedding similarity
        if not bridges and use_embedding_fallback and self.embedder:
            similarities = np.dot(
                theme_index(self.embedder), embed_query(self.embedder, query)
            )
            top_indices = np.argsort(similarities)[::-1][:max_bridges]

            for idx in top_indices:
                if similarities[idx] >= self.min_confidence:
                    bridges.append(QURANIC_THEMES[idx])
                    confidence[QURANIC_THEMES[idx]] = float(similarities[idx])

        # Unverified seed concepts as a last resort
        if not bridges:
            for entity in seed_entities[:max_bridges]:
                bridges.append(entity)
                confidence[entity] = 0.5

        return BridgeResult(
            bridges=bridges[:max_bridges],
            entities_found=entities_found,
            relationships_traversed=relationships,
            confidence_scores=confidence,
        )

    def generate_bridges_sync(
        self,
        query: str,
        max_bridges: int = 5,
    ) -> BridgeResult:
        """Synchronous wrapper for generate_bridges."""
        return run_sync(self.generate_bridges(query, max_bridges))
