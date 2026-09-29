"""Canonical verse numbering for the one-verse-per-line Quran corpus.

Every consumer (vector building, the vector index, graph ingestion) reads
verses through this module, so a retrieved passage can always be cited as
surah:ayah. The file must hold exactly one line per verse in mushaf order.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Union

Resolution = Literal["verse", "paragraph", "surah"]

# Verse counts of the 114 surahs (Hafs numbering), 6,236 verses in total.
SURAH_VERSE_COUNTS = (
    7, 286, 200, 176, 120, 165, 206, 75, 129, 109,
    123, 111, 43, 52, 99, 128, 111, 110, 98, 135,
    112, 78, 118, 64, 77, 227, 93, 88, 69, 60,
    34, 30, 73, 54, 45, 83, 182, 88, 75, 85,
    54, 53, 89, 59, 37, 35, 38, 29, 18, 45,
    60, 49, 62, 55, 78, 96, 29, 22, 24, 13,
    14, 11, 11, 18, 12, 12, 30, 52, 52, 44,
    28, 28, 20, 56, 40, 31, 50, 40, 46, 42,
    29, 19, 36, 25, 22, 17, 19, 26, 30, 20,
    15, 21, 11, 8, 8, 19, 5, 8, 8, 11,
    11, 8, 3, 9, 5, 4, 7, 3, 6, 3,
    5, 4, 5, 6,
)
TOTAL_VERSES = sum(SURAH_VERSE_COUNTS)


class CorpusError(ValueError):
    """The corpus does not match the canonical verse layout."""


@dataclass(frozen=True)
class Passage:
    """A contiguous run of verses within one surah."""

    surah: int
    first_ayah: int
    last_ayah: int
    text: str

    @property
    def ref(self) -> str:
        """Citation such as ``2:255`` or ``2:1-19``."""
        if self.first_ayah == self.last_ayah:
            return f"{self.surah}:{self.first_ayah}"
        return f"{self.surah}:{self.first_ayah}-{self.last_ayah}"

    def metadata(self, resolution: str, index: int) -> dict:
        return {
            "resolution": resolution,
            "index": index,
            "surah": self.surah,
            "ayah_start": self.first_ayah,
            "ayah_end": self.last_ayah,
            "ref": self.ref,
        }


def load_verses(path: Union[str, Path]) -> List[Passage]:
    """Read the corpus and number every verse, or raise CorpusError."""
    lines = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]
    if len(lines) != TOTAL_VERSES:
        raise CorpusError(
            f"{path}: expected {TOTAL_VERSES} verses, one per line in mushaf order, "
            f"found {len(lines)} non-empty lines"
        )
    verses = []
    position = 0
    for surah, count in enumerate(SURAH_VERSE_COUNTS, start=1):
        for ayah in range(1, count + 1):
            verses.append(Passage(surah, ayah, ayah, lines[position]))
            position += 1
    return verses


def group_passages(verses: List[Passage], resolution: Resolution, size: int = 19) -> List[Passage]:
    """Group verses into passages that never cross a surah boundary.

    ``verse`` keeps single verses, ``paragraph`` makes consecutive windows of
    ``size`` verses within each surah, and ``surah`` makes one passage per surah.
    """
    if resolution == "verse":
        return list(verses)
    if resolution not in ("paragraph", "surah"):
        raise ValueError(f"Unknown resolution: {resolution}")
    if type(size) is not int or size < 1:
        raise ValueError("Passage size must be a positive integer")
    by_surah: dict = {}
    for verse in verses:
        by_surah.setdefault(verse.surah, []).append(verse)
    passages = []
    for surah, members in by_surah.items():
        step = len(members) if resolution == "surah" else size
        for start in range(0, len(members), step):
            window = members[start:start + step]
            passages.append(
                Passage(
                    surah,
                    window[0].first_ayah,
                    window[-1].last_ayah,
                    " ".join(verse.text for verse in window),
                )
            )
    return passages
