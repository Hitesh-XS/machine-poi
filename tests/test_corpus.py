"""Canonical verse numbering, passage boundaries and citations."""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from machine_poi.corpus import (
    SURAH_VERSE_COUNTS,
    TOTAL_VERSES,
    CorpusError,
    Passage,
    group_passages,
    load_verses,
)

CORPUS = Path(__file__).parent.parent / "al-quran.txt"


@pytest.fixture(scope="module")
def verses():
    return load_verses(CORPUS)


def by_ref(items):
    return {item.ref: item.text for item in items}


def test_real_corpus_numbering_is_canonical(verses):
    refs = by_ref(verses)
    assert len(verses) == TOTAL_VERSES == 6236 and len(SURAH_VERSE_COUNTS) == 114
    assert refs["1:1"] == "بسم الله الرحمن الرحيم"
    assert refs["2:1"] == "الم"
    assert refs["114:6"] == "من الجنة والناس"


def test_short_verses_are_kept(verses):
    # A 10-character minimum used to drop 45 verses and shift later indices.
    refs = by_ref(verses)
    assert refs["20:30"] == "هارون أخي" and refs["114:2"] == "ملك الناس"
    assert sum(len(text) < 10 for text in refs.values()) == 45


def test_malformed_corpus_is_rejected(malformed_quran_path):
    with pytest.raises(CorpusError, match="found 10"):
        load_verses(malformed_quran_path)


@pytest.mark.parametrize("resolution", ["paragraph", "surah"])
def test_passages_stay_within_surahs_and_cover_every_verse(sample_quran_path, resolution):
    verses = load_verses(sample_quran_path)
    passages = group_passages(verses, resolution, size=19)
    covered = [
        f"{p.surah}:{a}" for p in passages for a in range(p.first_ayah, p.last_ayah + 1)
    ]
    assert covered == [v.ref for v in verses]
    for passage in passages:
        assert passage.text.split(" ")[1] == f"{passage.surah}:{passage.first_ayah}"
        assert passage.text.endswith(f"{passage.surah}:{passage.last_ayah}")


def test_passage_references(sample_quran_path):
    verses = load_verses(sample_quran_path)
    paragraphs = [p.ref for p in group_passages(verses, "paragraph", size=19)]
    assert paragraphs[:3] == ["1:1-7", "2:1-19", "2:20-38"]
    assert "2:286" in paragraphs  # 286 = 15 * 19 + 1; windows never cross into surah 3
    surahs = [p.ref for p in group_passages(verses, "surah")]
    assert len(surahs) == 114 and surahs[1] == "2:1-286" and surahs[-1] == "114:1-6"


def test_index_stores_references_and_queries_return_them(tmp_path, sample_embedding_dim):
    from unittest.mock import patch

    from machine_poi.knowledge_base import QuranKnowledgeBase
    from machine_poi.quran_embeddings import QuranEmbeddings

    passages = {
        "verse": [Passage(2, a, a, f"verse 2:{a}") for a in range(1, 6)],
        "paragraph": [Passage(2, 1, 5, "passage 2:1-5")],
        "surah": [Passage(2, 1, 286, "surah 2")],
    }
    with patch("machine_poi.knowledge_base.QuranEmbeddings") as factory:
        embedder = Mock(spec=QuranEmbeddings)
        embedder.model_id = "fake-embedder"
        embedder.embedding_dimension.return_value = sample_embedding_dim
        embedder.load_passages.side_effect = lambda path, chunk_by: passages[chunk_by]
        embedder.create_embeddings.side_effect = lambda texts, **kw: np.ones(
            (len(texts), sample_embedding_dim), dtype=np.float32
        )
        factory.return_value = embedder
        kb = QuranKnowledgeBase(persist_dir=str(tmp_path / "db"), device="cpu")
        kb.build_index()
        stored = kb.collections["verse"].get(ids=["verse_2:3"])
        assert stored["metadatas"][0] == {
            "resolution": "verse", "index": 2, "surah": 2,
            "ayah_start": 3, "ayah_end": 3, "ref": "2:3",
        }
        results = kb.query_multiresolution("q", n_results=2)
    assert {item["ref"] for item in results["verse"]} <= {f"2:{a}" for a in range(1, 6)}
    assert results["surah"][0]["ref"] == "2:1-286"


def test_mra_prompt_cites_retrieved_passages():
    from machine_poi.knowledge_base import QuranKnowledgeBase
    from machine_poi.steerer import QuranSteerer

    steerer = QuranSteerer(device="cpu")
    steerer.knowledge_base = Mock(spec=QuranKnowledgeBase)
    steerer.knowledge_base.query_multiresolution.return_value = {
        "verse": [{"content": "الله لا إله إلا هو", "ref": "2:255"}],
        "passage": [{"content": "passage text", "ref": "2:254-272"}],
        "surah": [{"content": "legacy item without a reference"}],
    }
    prompt, _ = steerer._mra_context("What is truth?", use_domain_bridges=False)
    assert "[2:255] الله" in prompt  # quoted context keeps the citation, readable
    assert "[2:254-272] passage text" in prompt
    assert "- legacy item without a reference" in prompt
