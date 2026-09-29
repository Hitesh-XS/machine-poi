"""Quranic theme vocabulary, keyword matching and a cached theme index.

Kept free of model dependencies so retrieval modules can share it without
importing the steering stack.
"""

import re
import threading
import weakref
from functools import lru_cache
from typing import Dict, Iterable, List

import numpy as np


# Domain bridge mappings: maps common concepts to Quranic themes
DOMAIN_BRIDGE_MAP: Dict[str, List[str]] = {
    # Technical/Programming domains
    "bug": ["correction", "improvement", "refinement", "fixing mistakes"],
    "debug": ["patience", "careful examination", "seeking truth"],
    "error": ["forgiveness", "learning from mistakes", "repentance"],
    "code": ["creation", "order", "structure", "wisdom"],
    "refactor": ["purification", "improvement", "renewal"],
    "optimize": ["excellence", "perfection", "ihsan"],
    "test": ["verification", "proof", "examination"],
    "deploy": ["trust in Allah", "tawakkul", "action after preparation"],

    # Teamwork/Social domains
    "team": ["unity", "brotherhood", "cooperation", "ummah"],
    "conflict": ["reconciliation", "peace-making", "patience"],
    "argue": ["respectful dialogue", "wisdom in speech", "reconciliation"],
    "collaborate": ["mutual help", "cooperation", "supporting one another"],
    "leadership": ["responsibility", "trust", "justice", "consultation"],
    "decision": ["consultation", "shura", "seeking guidance", "istikharah"],

    # Personal/Emotional domains
    "stress": ["patience", "sabr", "trust in Allah", "peace of heart"],
    "anxiety": ["remembrance of Allah", "tranquility", "tawakkul"],
    "failure": ["perseverance", "learning", "hope", "never despair"],
    "success": ["gratitude", "shukr", "humility", "continued effort"],
    "motivation": ["purpose", "intention", "seeking Allah's pleasure"],
    "fear": ["courage", "trust", "hope in Allah's mercy"],

    # Learning/Growth domains
    "learn": ["seeking knowledge", "wisdom", "reflection", "tadabbur"],
    "understand": ["contemplation", "insight", "divine guidance"],
    "teach": ["conveying truth", "patience", "wisdom", "example"],
    "growth": ["spiritual development", "self-improvement", "tarbiyah"],

    # General life domains
    "money": ["trust", "provision from Allah", "gratitude", "moderation"],
    "health": ["blessing", "patience in hardship", "gratitude"],
    "family": ["mercy", "compassion", "responsibility", "kindness to parents"],
    "time": ["value of time", "not wasting life", "preparation for hereafter"],
    "death": ["certainty", "preparation", "meeting Allah", "legacy"],
    "life": ["purpose", "test", "journey to Allah", "worship"],
}


# Curated Quranic themes for embedding-based auto-bridge generation
QURANIC_THEMES: List[str] = [
    # Core spiritual concepts
    "patience and perseverance (sabr)",
    "gratitude and thankfulness (shukr)",
    "trust and reliance on Allah (tawakkul)",
    "repentance and seeking forgiveness (tawbah)",
    "remembrance of Allah (dhikr)",
    "spiritual purification (tazkiyah)",
    "excellence in worship (ihsan)",
    "consciousness of Allah (taqwa)",
    
    # Moral virtues
    "honesty and truthfulness",
    "justice and fairness",
    "mercy and compassion",
    "humility and modesty",
    "generosity and charity",
    "kindness to parents and family",
    "fulfilling promises and trusts",
    "forgiving others",
    
    # Life guidance
    "dealing with hardship and trials",
    "hope and never despairing",
    "balance and moderation",
    "seeking knowledge and wisdom",
    "reflection and contemplation (tadabbur)",
    "taking responsibility",
    "preparing for the hereafter",
    "purpose and meaning of life",
    
    # Social relations
    "brotherhood and unity",
    "consultation and cooperation (shura)",
    "reconciliation and peace-making",
    "respectful dialogue",
    "supporting one another",
    "community (ummah)",
    
    # Work and action
    "striving with effort (jihad al-nafs)",
    "excellence in work",
    "fulfilling duties and obligations",
    "taking action after preparation",
    "persisting despite difficulties",
    "learning from mistakes",
    
    # Inner states
    "peace and tranquility of heart",
    "contentment and inner satisfaction",
    "overcoming fear and anxiety",
    "building confidence through faith",
    "finding strength in adversity",
]


@lru_cache(maxsize=None)
def _keyword_pattern(keyword: str) -> "re.Pattern[str]":
    """Match a keyword as a whole word, allowing common English inflections."""
    word = keyword.lower()
    stem = re.escape(word[:-1])
    if word.endswith("e"):
        body = f"{stem}(?:e|es|ed|ing|er|ers)"
    elif len(word) > 1 and word.endswith("y") and word[-2] not in "aeiou":
        body = f"{stem}(?:y|ies|ied)"
    else:
        last = re.escape(word[-1])
        body = f"{re.escape(word)}(?:{last}?(?:ed|ing|er|ers)|s|es)?"
    return re.compile(rf"\b{body}\b")


def matching_keywords(text: str, keywords: Iterable[str]) -> List[str]:
    """Return the keywords that occur as words in text, in keyword order.

    Substring matching produced bridges such as "terror" -> "error" and
    "latest" -> "test"; word boundaries prevent those matches.
    """
    lowered = text.lower()
    return [keyword for keyword in keywords if _keyword_pattern(keyword).search(lowered)]


_theme_indices: "weakref.WeakKeyDictionary[object, np.ndarray]" = weakref.WeakKeyDictionary()
_theme_lock = threading.Lock()


def theme_index(embedder) -> np.ndarray:
    """Normalized QURANIC_THEMES embeddings, computed once per embedder."""
    with _theme_lock:
        cached = _theme_indices.get(embedder)
    if cached is not None:
        return cached
    embeddings = np.asarray(
        embedder.create_embeddings(QURANIC_THEMES, show_progress=False), dtype=np.float32
    )
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    index = embeddings / (norms + 1e-8)
    with _theme_lock:
        _theme_indices[embedder] = index
    return index


def embed_query(embedder, query: str) -> np.ndarray:
    """Embed one query and L2-normalize it."""
    embedding = np.asarray(
        embedder.create_embeddings([query], show_progress=False)[0], dtype=np.float32
    )
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else embedding
