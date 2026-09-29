"""
Quran Text Embeddings Module

Creates semantic embeddings from Quranic verses using models like:
- Qwen3-Embedding-8B (Alibaba-NLP/gte-Qwen2-7B-instruct or similar)
- BGE-M3 (BAAI/bge-m3)
"""

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Union, List, Dict, Literal
import numpy as np
import torch

from .config import EMBEDDING_MODELS, STEERING_DEFAULTS
from .corpus import Passage, Resolution, group_passages, load_verses

# Setup module logger
logger = logging.getLogger("machine_poi.quran_embeddings")


class EmbeddingError(Exception):
    """Base exception for embedding-related errors."""
    pass


class QuranFileError(EmbeddingError):
    """Raised when Quran file is invalid or missing."""
    pass


DEFAULT_CORPUS = "al-quran.txt"


def resolve_corpus_path(file_path: Union[str, Path]) -> Path:
    """Resolve a corpus path; only the default name falls back to the checkout.

    Any other missing path is an error, so a typo cannot silently load a
    different corpus.
    """
    path = Path(file_path)
    if path.exists():
        return path
    if path == Path(DEFAULT_CORPUS):
        fallback = Path(__file__).resolve().parent.parent / DEFAULT_CORPUS
        if fallback.exists():
            return fallback
    raise QuranFileError(f"Quran text file not found: {file_path}")


class LRUCache:
    """Simple LRU cache for embeddings."""
    
    def __init__(self, max_size: int = 1000):
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.max_size = max_size
    
    def get(self, key: str) -> Optional[np.ndarray]:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value: np.ndarray) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
            self.cache[key] = value
    
    def clear(self) -> None:
        self.cache.clear()
    
    def __len__(self) -> int:
        return len(self.cache)



class QuranEmbeddings:
    """
    Creates and manages embeddings from Quran text using various embedding models.

    Supports:
    - BAAI/bge-m3: Multilingual embeddings with strong Arabic support
    - Alibaba-NLP/gte-Qwen2-7B-instruct: Large-scale instruction-tuned embeddings
    """

    SUPPORTED_MODELS = {alias: spec["hf_path"] for alias, spec in EMBEDDING_MODELS.items()}

    def __init__(
        self,
        model_name: str = "paraphrase-minilm",
        device: Optional[str] = None,
        use_fp16: bool = True,
        max_length: int = 512,
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
    ):
        """
        Initialize the Quran embeddings generator.

        Args:
            model_name: Name of the embedding model to use
            device: Device to run on (cuda/cpu/mps)
            use_fp16: Use half precision for memory efficiency
            max_length: Maximum sequence length for embeddings
        """
        import re
        if trust_remote_code and not re.fullmatch(r"[0-9a-fA-F]{40}", revision or ""):
            raise ValueError("Remote code requires a pinned commit revision")
        self.revision = revision
        self.trust_remote_code = trust_remote_code
        self.model_name = model_name
        self.max_length = max_length
        self.use_fp16 = use_fp16

        # Determine device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.model = None
        self.tokenizer = None
        self._embeddings_cache = LRUCache(max_size=STEERING_DEFAULTS.max_embedding_cache_size)

    def load_model(self) -> None:
        """Load the embedding model."""
        from sentence_transformers import SentenceTransformer

        model_path = self.SUPPORTED_MODELS.get(self.model_name, self.model_name)
        logger.info(f"Loading embedding model: {model_path}")

        self.model = SentenceTransformer(
            model_path, device=self.device, revision=self.revision,
            trust_remote_code=self.trust_remote_code,
        )

        if self.use_fp16 and self.device != "cpu":
            self.model = self.model.half()

        logger.info(f"Model loaded on {self.device}")

    def load_passages(
        self,
        file_path: Union[str, Path] = "al-quran.txt",
        chunk_by: Resolution = "verse",
    ) -> List[Passage]:
        """
        Load the corpus as cited passages that never cross a surah boundary.

        Args:
            file_path: One-verse-per-line corpus with 6,236 lines. The default
                name also resolves relative to the repository checkout.
            chunk_by: "verse", "paragraph" (fixed windows within a surah) or "surah"

        Raises:
            QuranFileError: If the file is missing or unreadable
            CorpusError: If the file does not have the canonical verse layout
        """
        file_path = resolve_corpus_path(file_path)
        try:
            verses = load_verses(file_path)
        except OSError as e:
            raise QuranFileError(f"Failed to read Quran file: {e}")
        passages = group_passages(
            verses, chunk_by, STEERING_DEFAULTS.paragraph_verse_count
        )
        logger.info(f"Loaded {len(passages)} passages from Quran ({chunk_by} mode)")
        return passages

    def load_quran_text(
        self,
        file_path: Union[str, Path] = "al-quran.txt",
        chunk_by: Resolution = "verse",
        min_length: int = 0,
    ) -> List[str]:
        """
        Load passage texts in canonical order.

        Every verse is kept by default, so list positions match verse order;
        ``min_length`` drops shorter chunks and breaks that correspondence.
        """
        return [
            passage.text
            for passage in self.load_passages(file_path, chunk_by)
            if len(passage.text) >= min_length
        ]

    def create_embeddings(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Create embeddings for a list of texts.

        Args:
            texts: List of text strings to embed
            batch_size: Batch size for encoding
            normalize: Whether to L2 normalize embeddings
            show_progress: Show progress bar

        Returns:
            Numpy array of embeddings [num_texts, embedding_dim]
        """
        if self.model is None:
            self.load_model()

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )

        return embeddings

    def create_quran_embeddings(
        self,
        file_path: Union[str, Path] = "al-quran.txt",
        chunk_by: Literal["verse", "surah", "paragraph"] = "verse",
        batch_size: int = 32,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Create embeddings for the entire Quran text.

        Args:
            file_path: Path to Quran text
            chunk_by: How to chunk the text
            batch_size: Batch size for encoding
            save_path: Optional path to save embeddings

        Returns:
            Dictionary with 'embeddings', 'texts', and 'mean_embedding'
        """
        texts = self.load_quran_text(file_path, chunk_by=chunk_by)
        embeddings = self.create_embeddings(texts, batch_size=batch_size)

        # Compute mean embedding (the "Quran vector")
        mean_embedding = np.mean(embeddings, axis=0)
        mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)

        result = {
            "embeddings": embeddings,
            "texts": texts,
            "mean_embedding": mean_embedding,
            "model_name": self.model_name,
            "chunk_by": chunk_by,
        }

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                save_path,
                embeddings=embeddings,
                mean_embedding=mean_embedding,
                model_name=np.array(self.model_name),
                chunk_by=np.array(chunk_by),
            )
            # Save texts separately (numpy doesn't handle variable-length strings well)
            with open(save_path.with_suffix(".texts.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(texts))
            logger.info(f"Saved embeddings to {save_path}")

        return result

    def load_cached_embeddings(
        self,
        load_path: Union[str, Path],
    ) -> Dict[str, np.ndarray]:
        """Load previously saved embeddings."""
        load_path = Path(load_path)
        with np.load(load_path, allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}

        texts = []
        texts_path = load_path.with_suffix(".texts.txt")
        if texts_path.exists():
            with open(texts_path, "r", encoding="utf-8") as f:
                texts = f.read().split("\n")

        return {
            "embeddings": data["embeddings"],
            "mean_embedding": data["mean_embedding"],
            "texts": texts,
            "model_name": str(data.get("model_name", "unknown")),
            "chunk_by": str(data.get("chunk_by", "unknown")),
        }

    def get_semantic_clusters(
        self,
        embeddings: np.ndarray,
        n_clusters: int = 10,
    ) -> Dict[str, np.ndarray]:
        """
        Cluster embeddings to find semantic themes.

        Args:
            embeddings: The embeddings array
            n_clusters: Number of clusters

        Returns:
            Dictionary with cluster centers and labels
        """
        from scipy.cluster.vq import kmeans2

        centers, labels = kmeans2(embeddings.astype(np.float64), n_clusters, minit="++")

        # Normalize cluster centers
        centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)

        return {
            "centers": centers.astype(np.float32),
            "labels": labels,
            "n_clusters": n_clusters,
        }
