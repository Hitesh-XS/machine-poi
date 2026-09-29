"""A persisted vector index must match the embedder, corpus and schema in use."""

import sys
from unittest.mock import Mock, patch

import chromadb
import numpy as np
import pytest

from machine_poi import cli
from machine_poi.corpus import Passage
from machine_poi.knowledge_base import QuranKnowledgeBase, StaleIndexError
from machine_poi.quran_embeddings import QuranEmbeddings

PASSAGES = {
    "verse": [Passage(1, a, a, f"verse 1:{a}") for a in range(1, 8)],
    "paragraph": [Passage(1, 1, 7, "passage 1:1-7")],
    "surah": [Passage(1, 1, 7, "surah 1")],
}


def embedder(model_id="model-a", dim=8):
    fake = Mock(spec=QuranEmbeddings)
    fake.model_id = model_id
    fake.embedding_dimension.return_value = dim
    fake.load_passages.side_effect = lambda path, chunk_by: PASSAGES[chunk_by]
    fake.create_embeddings.side_effect = lambda texts, **kw: np.ones(
        (len(texts), dim), dtype=np.float32
    )
    return fake


def knowledge_base(tmp_path, **kwargs):
    return QuranKnowledgeBase(persist_dir=str(tmp_path / "db"), embedder=embedder(**kwargs))


def test_switching_embedding_model_fails_clearly_until_rebuilt(tmp_path):
    knowledge_base(tmp_path).build_index()
    switched = knowledge_base(tmp_path, model_id="model-b")
    with pytest.raises(StaleIndexError, match="embedding_model: index has 'model-a', current is 'model-b'.*--rebuild"):
        switched.query_multiresolution("q")
    with pytest.raises(StaleIndexError):
        switched.build_index()

    switched.build_index(rebuild=True)
    assert switched.query_multiresolution("q", n_results=1)["verse"][0]["ref"].startswith("1:")
    assert switched.collections["verse"].metadata["embedding_model"] == "model-b"


def test_corpus_change_is_detected(tmp_path, sample_quran_path):
    knowledge_base(tmp_path).build_index()
    other = QuranKnowledgeBase(
        persist_dir=str(tmp_path / "db"), embedder=embedder(), quran_path=sample_quran_path
    )
    with pytest.raises(StaleIndexError, match="corpus_sha256"):
        other.verify()


def test_legacy_index_without_identity_requires_rebuild(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path / "db"))
    legacy = client.get_or_create_collection("quran_verses", metadata={"hnsw:space": "cosine"})
    legacy.add(ids=["verse_0"], embeddings=[[1.0] * 8], documents=["old"])
    with pytest.raises(StaleIndexError, match="schema_version: index has None, current is 2"):
        knowledge_base(tmp_path).query_multiresolution("q")


def test_matching_index_is_reused_without_reembedding(tmp_path):
    knowledge_base(tmp_path).build_index()
    again = knowledge_base(tmp_path)
    again.build_index()
    again.embedder.create_embeddings.assert_not_called()


def test_steerer_shares_its_loaded_embedder_with_the_index():
    from machine_poi.steerer import QuranSteerer

    steerer = QuranSteerer(device="cpu")
    steerer.embedder = embedder()
    with patch("machine_poi.steerer.QuranKnowledgeBase") as factory:
        steerer.initialize_knowledge_base()
    kwargs = factory.call_args.kwargs
    assert kwargs["embedder"] is steerer.embedder
    assert kwargs["quran_path"] == steerer.quran_path


@pytest.mark.parametrize("rebuild", [False, True])
def test_cli_rebuild_flag_and_stale_index_exit(monkeypatch, rebuild):
    steerer = Mock()
    monkeypatch.setattr(cli, "QuranSteerer", Mock(return_value=steerer))
    argv = ["machine-poi", "--init-db"] + (["--rebuild"] if rebuild else [])
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    assert steerer.knowledge_base.build_index.call_args.kwargs == {"rebuild": rebuild}

    steerer.knowledge_base.build_index.side_effect = StaleIndexError("model changed")
    with pytest.raises(SystemExit, match="Stale vector index: model changed"):
        cli.main()
