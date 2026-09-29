"""
Knowledge Base Module using ChromaDB.

Manages multi-resolution indexing of the Quran:
1. Micro: Individual Verses
2. Meso: Passages (Thematic groups)
3. Macro: Surahs (Chapters)
"""

import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union
import chromadb

from .quran_embeddings import QuranEmbeddings, resolve_corpus_path

from .config import STEERING_DEFAULTS
from .corpus import Passage

# Setup logger
logger = logging.getLogger("machine_poi.knowledge_base")


class KnowledgeBaseError(Exception):
    """Base exception for knowledge base errors."""
    pass


class EmptyCollectionError(KnowledgeBaseError):
    """Raised when querying an empty collection."""
    pass


class StaleIndexError(KnowledgeBaseError):
    """The persisted index was built with another embedder, corpus or schema."""


# Version 2 added verse references to documents and metadata.
INDEX_SCHEMA_VERSION = 2


class QuranKnowledgeBase:
    """
    Manages Quranic knowledge in ChromaDB with multi-resolution support.
    """

    COLLECTION_NAMES = {
        "verse": "quran_verses",
        "passage": "quran_passages",
        "surah": "quran_surahs",
    }

    def __init__(
        self,
        persist_dir: str = "quran_db",
        embedding_model_name: str = "paraphrase-minilm",
        device: Optional[str] = None,
        embedder: Optional[QuranEmbeddings] = None,
        quran_path: Union[str, Path] = "al-quran.txt",
    ):
        """
        Initialize the knowledge base.

        Args:
            persist_dir: Directory to store ChromaDB data
            embedding_model_name: Name of embedding model to use
            device: Computation device
            embedder: Loaded embedder to share instead of loading another copy
            quran_path: Corpus whose hash identifies the index contents
        """
        self.persist_dir = persist_dir
        self.client = chromadb.PersistentClient(path=persist_dir)

        if embedder is None:
            embedder = QuranEmbeddings(model_name=embedding_model_name, device=device)
            embedder.load_model()
        self.embedder = embedder
        self._set_corpus(quran_path)

        # Collections store the identity they were created with; reopening an
        # existing collection keeps its stored metadata.
        self.collections = {
            resolution: self.client.get_or_create_collection(
                name=name, metadata=self._collection_metadata()
            )
            for resolution, name in self.COLLECTION_NAMES.items()
        }

    def _set_corpus(self, quran_path: Union[str, Path]) -> None:
        self.quran_path = resolve_corpus_path(quran_path)
        self.identity = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "embedding_model": self.embedder.model_id,
            "embedding_dim": self.embedder.embedding_dimension(),
            "corpus_sha256": hashlib.sha256(self.quran_path.read_bytes()).hexdigest(),
        }

    def _collection_metadata(self) -> dict:
        return {"hnsw:space": "cosine", **self.identity}

    def _mismatches(self, collection) -> Dict[str, tuple]:
        stored = collection.metadata or {}
        return {
            key: (stored.get(key), value)
            for key, value in self.identity.items()
            if stored.get(key) != value
        }

    def _stale_error(self, resolution: str, mismatches: Dict[str, tuple]) -> StaleIndexError:
        def show(value):
            return value[:12] if isinstance(value, str) and len(value) == 64 else value

        details = "; ".join(
            f"{key}: index has {show(stored)!r}, current is {show(current)!r}"
            for key, (stored, current) in mismatches.items()
        )
        return StaleIndexError(
            f"The {resolution} index in {self.persist_dir} does not match the current "
            f"configuration ({details}). Rebuild it with `machine-poi --init-db "
            f"--rebuild` or build_index(rebuild=True)."
        )

    def verify(self) -> None:
        """Raise StaleIndexError if a populated collection has a different identity."""
        for resolution, collection in self.collections.items():
            mismatches = self._mismatches(collection)
            if mismatches and collection.count() > 0:
                raise self._stale_error(resolution, mismatches)

    def _recreate(self, resolution: str) -> None:
        name = self.COLLECTION_NAMES[resolution]
        self.client.delete_collection(name)
        self.collections[resolution] = self.client.create_collection(
            name=name, metadata=self._collection_metadata()
        )

    def build_index(
        self, quran_path: Optional[Union[str, Path]] = None, rebuild: bool = False
    ) -> None:
        """
        Build the index, skipping collections already built with this identity.

        A populated collection built with a different embedding model, corpus
        or schema raises StaleIndexError unless ``rebuild`` is set, in which
        case every collection is rebuilt.
        """
        if quran_path is not None:
            self._set_corpus(quran_path)
        logger.info("Building Knowledge Base Index...")

        # Verses (micro), passages within a surah (meso) and whole surahs (macro)
        for resolution, chunk_by, batch_size in (
            ("verse", "verse", STEERING_DEFAULTS.verse_index_batch_size),
            ("passage", "paragraph", STEERING_DEFAULTS.passage_index_batch_size),
            ("surah", "surah", STEERING_DEFAULTS.surah_index_batch_size),
        ):
            collection = self.collections[resolution]
            mismatches = self._mismatches(collection)
            populated = collection.count() > 0
            if populated and mismatches and not rebuild:
                raise self._stale_error(resolution, mismatches)
            if rebuild or mismatches:
                self._recreate(resolution)
            elif populated:
                logger.info(f"Collection {resolution} is already built. Skipping.")
                continue
            passages = self.embedder.load_passages(self.quran_path, chunk_by=chunk_by)
            self._index_collection(resolution, passages, batch_size=batch_size)

        logger.info("Indexing complete!")

    def _index_collection(self, resolution: str, passages: List[Passage], batch_size: int) -> None:
        """Helper to index a specific resolution."""
        collection = self.collections[resolution]

        logger.info(f"Indexing {len(passages)} {resolution}s...")

        # Generate embeddings in batches
        embeddings = self.embedder.create_embeddings(
            [passage.text for passage in passages], batch_size=batch_size
        )

        # Add to Chroma in batches to avoid message size limits. IDs and
        # metadata carry the surah:ayah reference for citation.
        total = len(passages)
        for i in range(0, total, batch_size):
            batch = passages[i:i + batch_size]
            collection.add(
                documents=[passage.text for passage in batch],
                embeddings=embeddings[i:i + batch_size].tolist(),
                ids=[f"{resolution}_{passage.ref}" for passage in batch],
                metadatas=[
                    passage.metadata(resolution, i + offset)
                    for offset, passage in enumerate(batch)
                ],
            )

    def query_multiresolution(
        self,
        query_text: str,
        n_results: int = 3,
        include_embeddings: bool = False,
    ) -> Dict[str, List[Dict]]:
        """
        Query all resolutions simultaneously.

        Args:
            query_text: The query string
            n_results: Number of results per resolution
            include_embeddings: Whether to include embeddings in results (for dynamic steering)

        Returns:
            Dict with 'verse', 'passage', 'surah' results.
        """
        self.verify()
        # Embed query
        query_embedding = self.embedder.create_embeddings([query_text])[0].tolist()

        results = {}
        for res_name, collection in self.collections.items():
            # Adjust n_results for macro levels (fewer surahs needed)
            k = n_results
            if res_name == "surah":
                k = max(1, n_results // 3)

            # Include embeddings if requested
            include_fields = ["documents", "metadatas", "distances"]
            if include_embeddings:
                include_fields.append("embeddings")

            response = collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=include_fields
            )

            # Formatter
            formatted = []
            if response["documents"]:
                docs = response["documents"][0]
                metas = response["metadatas"][0]
                dists = response["distances"][0]
                embeds = response.get("embeddings", [[None] * len(docs)])[0] if include_embeddings else [None] * len(docs)

                for doc, meta, dist, emb in zip(docs, metas, dists, embeds):
                    item = {
                        "content": doc,
                        "ref": (meta or {}).get("ref"),
                        "metadata": meta,
                        "distance": dist,
                        "score": 1.0 - dist  # Cosine distance to similarity
                    }
                    if include_embeddings and emb is not None:
                        item["embedding"] = emb
                    formatted.append(item)
            results[res_name] = formatted

        return results

    def query_with_bridges(
        self,
        original_query: str,
        bridge_queries: List[str],
        n_results: int = 3,
        include_embeddings: bool = False,
    ) -> Dict[str, List[Dict]]:
        """
        Query using both original query and domain bridge queries.

        Combines results from the original query and bridge queries,
        de-duplicating and re-ranking by best score.

        Args:
            original_query: The user's original query
            bridge_queries: List of domain bridge queries
            n_results: Number of results per resolution
            include_embeddings: Whether to include embeddings

        Returns:
            Dict with 'verse', 'passage', 'surah' results (merged and ranked).
        """
        all_queries = [original_query] + bridge_queries

        # Collect all results
        merged_results = {"verse": {}, "passage": {}, "surah": {}}

        for query in all_queries:
            results = self.query_multiresolution(
                query,
                n_results=n_results,
                include_embeddings=include_embeddings
            )

            for res_name, items in results.items():
                for item in items:
                    doc_id = item.get("ref") or item["metadata"].get("index", item["content"][:50])
                    # Keep the highest scoring occurrence
                    if doc_id not in merged_results[res_name] or item["score"] > merged_results[res_name][doc_id]["score"]:
                        merged_results[res_name][doc_id] = item

        # Convert back to list and sort by score
        final_results = {}
        for res_name, items_dict in merged_results.items():
            sorted_items = sorted(items_dict.values(), key=lambda x: x["score"], reverse=True)
            final_results[res_name] = sorted_items[:n_results]

        return final_results

    def get_weighted_embedding(self, query_results: Dict[str, List[Dict]]) -> Dict[str, float]:
        """
        Calculate a comprehensive strategy for steering based on retrieved results.
        This is a placeholder for more advanced logic.
        """
        # Not used for steering directly yet, but helpful for logic
        pass
