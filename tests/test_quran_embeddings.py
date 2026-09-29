"""
Tests for machine_poi/quran_embeddings.py

Tests the QuranEmbeddings class for:
- Text loading and chunking
- Embedding creation
- Caching and loading
- Semantic clustering
"""

import pytest
import numpy as np


class TestQuranEmbeddingsInit:
    """Test QuranEmbeddings initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        from machine_poi.quran_embeddings import QuranEmbeddings
        
        embedder = QuranEmbeddings()
        
        assert embedder.model_name == "paraphrase-minilm"
        assert embedder.max_length == 512
        assert embedder.use_fp16 is True
        assert embedder.model is None  # Not loaded yet

    def test_init_with_custom_model(self):
        """Test initialization with custom model name."""
        from machine_poi.quran_embeddings import QuranEmbeddings
        
        embedder = QuranEmbeddings(model_name="multilingual-e5")
        
        assert embedder.model_name == "multilingual-e5"

    def test_init_device_selection_cpu(self):
        """Test explicit CPU device selection."""
        from machine_poi.quran_embeddings import QuranEmbeddings
        
        embedder = QuranEmbeddings(device="cpu")
        
        assert embedder.device == "cpu"

    def test_supported_models_mapping(self):
        """Test that supported models are correctly mapped."""
        from machine_poi.quran_embeddings import QuranEmbeddings
        
        # Match current SUPPORTED_MODELS in quran_embeddings.py
        expected_models = ["bge-m3", "qwen-embedding", "multilingual-e5", "paraphrase-mpnet"]
        
        for model in expected_models:
            assert model in QuranEmbeddings.SUPPORTED_MODELS


class TestTextLoading:
    """Test Quran text loading and chunking."""

    def test_load_quran_text_verse_chunking(self, sample_quran_path):
        """Test loading text with verse-level chunking."""
        from machine_poi.quran_embeddings import QuranEmbeddings
        
        embedder = QuranEmbeddings()
        chunks = embedder.load_quran_text(sample_quran_path, chunk_by="verse")
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(c, str) for c in chunks)

    def test_load_quran_text_paragraph_chunking(self, sample_quran_path):
        """Test loading text with paragraph-level chunking."""
        from machine_poi.quran_embeddings import QuranEmbeddings
        
        embedder = QuranEmbeddings()
        chunks = embedder.load_quran_text(sample_quran_path, chunk_by="paragraph")
        
        # Paragraphs should be fewer than verses
        verse_chunks = embedder.load_quran_text(sample_quran_path, chunk_by="verse")
        assert len(chunks) <= len(verse_chunks)

    def test_load_quran_text_surah_chunking(self, sample_quran_path):
        """Test loading text with surah-level chunking."""
        from machine_poi.quran_embeddings import QuranEmbeddings
        
        embedder = QuranEmbeddings()
        chunks = embedder.load_quran_text(sample_quran_path, chunk_by="surah")
        
        # Surahs should be fewer or equal to verses (the smallest unit)
        # Note: With small test files, surahs may outnumber paragraphs if verses
        # span multiple surahs but don't reach the paragraph grouping threshold
        verse_chunks = embedder.load_quran_text(sample_quran_path, chunk_by="verse")
        assert len(chunks) <= len(verse_chunks)

    def test_load_quran_text_min_length_filter(self, sample_quran_path):
        """Test that short chunks are filtered out."""
        from machine_poi.quran_embeddings import QuranEmbeddings
        
        embedder = QuranEmbeddings()
        
        # With high min_length, should filter out short verses
        chunks_strict = embedder.load_quran_text(
            sample_quran_path, chunk_by="verse", min_length=100
        )
        chunks_lenient = embedder.load_quran_text(
            sample_quran_path, chunk_by="verse", min_length=5
        )
        
        assert len(chunks_strict) <= len(chunks_lenient)

    def test_missing_explicit_path_raises_instead_of_loading_another_corpus(self, tmp_path):
        """Only the default corpus name falls back to the checkout's copy."""
        from machine_poi.quran_embeddings import QuranEmbeddings, QuranFileError

        embedder = QuranEmbeddings()
        with pytest.raises(QuranFileError):
            embedder.load_quran_text(tmp_path / "nonexistent.txt")

    def test_default_corpus_name_resolves_from_another_directory(self, tmp_path, monkeypatch):
        from machine_poi.quran_embeddings import QuranEmbeddings

        monkeypatch.chdir(tmp_path)
        assert len(QuranEmbeddings().load_quran_text()) == 6236

    def test_malformed_corpus_is_rejected(self, malformed_quran_path):
        from machine_poi.corpus import CorpusError
        from machine_poi.quran_embeddings import QuranEmbeddings

        with pytest.raises(CorpusError, match="expected 6236 verses.*found 10"):
            QuranEmbeddings().load_quran_text(malformed_quran_path)


class TestEmbeddingCreation:
    """Test embedding creation functionality."""

    def test_create_embeddings_returns_correct_shape(
        self, mock_sentence_transformer, sample_embedding_dim
    ):
        """Test that embeddings have correct shape."""
        from machine_poi.quran_embeddings import QuranEmbeddings
        
        embedder = QuranEmbeddings()
        embedder.model = mock_sentence_transformer
        
        texts = ["text one", "text two", "text three"]
        embeddings = embedder.create_embeddings(texts)
        
        assert embeddings.shape == (3, sample_embedding_dim)

    def test_create_embeddings_normalized(
        self, mock_sentence_transformer, sample_embedding_dim
    ):
        """Test that embeddings are L2 normalized."""
        from machine_poi.quran_embeddings import QuranEmbeddings
        
        embedder = QuranEmbeddings()
        embedder.model = mock_sentence_transformer
        
        texts = ["sample text"]
        embeddings = embedder.create_embeddings(texts, normalize=True)
        
        # Check L2 norm is approximately 1
        norm = np.linalg.norm(embeddings[0])
        assert abs(norm - 1.0) < 1e-5

    def test_create_quran_embeddings_returns_dict(
        self, mock_sentence_transformer, sample_quran_path
    ):
        """Test that create_quran_embeddings returns proper dict."""
        from machine_poi.quran_embeddings import QuranEmbeddings
        
        embedder = QuranEmbeddings()
        embedder.model = mock_sentence_transformer
        
        result = embedder.create_quran_embeddings(
            file_path=sample_quran_path,
            chunk_by="verse",
        )
        
        assert "embeddings" in result
        assert "texts" in result
        assert "mean_embedding" in result
        assert "model_name" in result
        assert "chunk_by" in result

    def test_create_quran_embeddings_mean_normalized(
        self, mock_sentence_transformer, sample_quran_path
    ):
        """Test that mean embedding is normalized."""
        from machine_poi.quran_embeddings import QuranEmbeddings
        
        embedder = QuranEmbeddings()
        embedder.model = mock_sentence_transformer
        
        result = embedder.create_quran_embeddings(
            file_path=sample_quran_path,
            chunk_by="verse",
        )
        
        mean_norm = np.linalg.norm(result["mean_embedding"])
        assert abs(mean_norm - 1.0) < 1e-5


class TestCaching:
    """Test embedding caching and loading."""

    def test_save_and_load_embeddings(
        self, mock_sentence_transformer, sample_quran_path, temp_cache_dir
    ):
        """Test that embeddings can be saved and loaded."""
        from machine_poi.quran_embeddings import QuranEmbeddings
        
        embedder = QuranEmbeddings()
        embedder.model = mock_sentence_transformer
        
        save_path = temp_cache_dir / "test_embeddings.npz"
        
        # Create and save
        result = embedder.create_quran_embeddings(
            file_path=sample_quran_path,
            save_path=save_path,
        )
        
        # Load
        loaded = embedder.load_cached_embeddings(save_path)
        
        np.testing.assert_array_almost_equal(
            result["embeddings"], loaded["embeddings"]
        )
        np.testing.assert_array_almost_equal(
            result["mean_embedding"], loaded["mean_embedding"]
        )

    def test_texts_saved_separately(
        self, mock_sentence_transformer, sample_quran_path, temp_cache_dir
    ):
        """Test that texts are saved in a separate file."""
        from machine_poi.quran_embeddings import QuranEmbeddings
        
        embedder = QuranEmbeddings()
        embedder.model = mock_sentence_transformer
        
        save_path = temp_cache_dir / "test_embeddings.npz"
        
        embedder.create_quran_embeddings(
            file_path=sample_quran_path,
            save_path=save_path,
        )
        
        texts_path = save_path.with_suffix(".texts.txt")
        assert texts_path.exists()


class TestSemanticClustering:
    """Test semantic clustering functionality."""

    def test_get_semantic_clusters(self, sample_embeddings):
        """Test that clustering returns correct structure."""
        from machine_poi.quran_embeddings import QuranEmbeddings
        
        embedder = QuranEmbeddings()
        n_clusters = 3
        
        result = embedder.get_semantic_clusters(sample_embeddings, n_clusters=n_clusters)
        
        assert "centers" in result
        assert "labels" in result
        assert "n_clusters" in result
        assert result["n_clusters"] == n_clusters
        assert len(result["labels"]) == len(sample_embeddings)

    def test_cluster_centers_normalized(self, sample_embeddings):
        """Test that cluster centers are normalized."""
        from machine_poi.quran_embeddings import QuranEmbeddings
        
        embedder = QuranEmbeddings()
        result = embedder.get_semantic_clusters(sample_embeddings, n_clusters=3)
        
        for center in result["centers"]:
            norm = np.linalg.norm(center)
            assert abs(norm - 1.0) < 1e-4


class TestModelLoading:
    """Test model loading (integration-style, may be slow)."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_load_model_bge_m3(self):
        """Test loading the bge-m3 model."""
        from machine_poi.quran_embeddings import QuranEmbeddings
        
        embedder = QuranEmbeddings(model_name="bge-m3", device="cpu")
        embedder.load_model()
        
        assert embedder.model is not None

    @pytest.mark.slow
    @pytest.mark.integration
    def test_encode_with_real_model(self):
        """Test encoding with a real model."""
        from machine_poi.quran_embeddings import QuranEmbeddings
        
        embedder = QuranEmbeddings(model_name="bge-m3", device="cpu")
        embedder.load_model()
        
        texts = ["بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"]
        embeddings = embedder.create_embeddings(texts)
        
        assert embeddings.shape[0] == 1
        assert embeddings.shape[1] > 0
