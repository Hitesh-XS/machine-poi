"""Neutral control texts for contrastive steering vectors.

A contrast between Quran verses and neutral text should differ in content,
not in language. The default control is Modern Standard Arabic written for
this project; the short English set is kept for explicit cross-language runs.
"""

import hashlib
from importlib import resources
from typing import Iterable, List

NEUTRAL_ENGLISH = [
    "The weather today is mild.",
    "Numbers are mathematical concepts.",
    "Water is composed of hydrogen and oxygen.",
    "Computers process information.",
    "Colors are perceived differently.",
    "Sound travels through air.",
    "Plants need sunlight to grow.",
    "Time passes continuously.",
    "Objects have mass and volume.",
    "Languages have grammar rules.",
]


def unique_texts(texts: Iterable[str]) -> List[str]:
    """Strip texts and drop empty lines and repeats, keeping first-seen order."""
    return list(dict.fromkeys(text.strip() for text in texts if text.strip()))


def neutral_texts(language: str = "ar") -> List[str]:
    """Deduplicated neutral sentences: "ar" (default) or "en"."""
    if language == "en":
        return unique_texts(NEUTRAL_ENGLISH)
    if language != "ar":
        raise ValueError(f"No neutral control set for language {language!r}")
    content = (
        resources.files("machine_poi")
        .joinpath("data/neutral_arabic.txt")
        .read_text(encoding="utf-8")
    )
    return unique_texts(line for line in content.splitlines() if not line.startswith("#"))


def texts_sha256(texts: Iterable[str]) -> str:
    """Order-sensitive hash of a text set, for cache and result metadata."""
    return hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()


def calibration_texts() -> List[str]:
    """Neutral sentences for dose calibration: the English set and ten Arabic ones."""
    return neutral_texts("en") + neutral_texts("ar")[:10]
