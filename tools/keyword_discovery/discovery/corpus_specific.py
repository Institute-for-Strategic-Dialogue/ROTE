"""Corpus-specific vocabulary discovery.

Finds terms that are distinctive to THIS corpus compared to general English,
following the intuition from Monroe et al. (2008) "Fightin' Words".

Three signals:
  1. **Non-dictionary terms** — words absent from a standard English lexicon
     are likely neologisms, jargon, loanwords, or in-group vocabulary
     (e.g. "goyslop", "khazarian", "wwg1wga", "looksmaxxing").
  2. **Productive affixes** — morphological roots that generate multiple
     novel corpus terms (e.g. "goy-" → goyim, goyslop, goybar).
  3. **Seed-adjacent non-English** — non-dictionary terms that co-occur
     with seeds get an additional boost.

This module is domain-agnostic: it requires no dictionaries, no topic
knowledge, and works for any subculture's specific vocabulary.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict

import numpy as np

from ..corpus import Corpus, get_seed_doc_mask
from ..models import Candidate
from .base import DiscoveryModule

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# English reference lexicon (loaded lazily from NLTK)
# ---------------------------------------------------------------------------
_ENGLISH_WORDS: set[str] | None = None


def _load_english_words() -> set[str]:
    """Load a comprehensive set of English words including inflected forms."""
    global _ENGLISH_WORDS
    if _ENGLISH_WORDS is not None:
        return _ENGLISH_WORDS

    try:
        from nltk.corpus import words as nltk_words
        base = {w.lower() for w in nltk_words.words()}
    except LookupError:
        import nltk
        nltk.download("words", quiet=True)
        from nltk.corpus import words as nltk_words
        base = {w.lower() for w in nltk_words.words()}

    # Generate common inflected forms (NLTK only has base forms)
    inflected: set[str] = set()
    for w in base:
        if len(w) < 2:
            continue
        inflected.add(w + "s")       # plurals, 3rd person
        inflected.add(w + "es")
        inflected.add(w + "ed")      # past tense
        inflected.add(w + "ing")     # gerund
        inflected.add(w + "er")      # comparative / agent
        inflected.add(w + "ers")
        inflected.add(w + "ly")      # adverbs
        inflected.add(w + "ness")    # nominalisation
        inflected.add(w + "ment")
        inflected.add(w + "tion")
        inflected.add(w + "ation")
        if w.endswith("e"):
            inflected.add(w[:-1] + "ing")   # make→making
            inflected.add(w[:-1] + "ed")    # use→used
            inflected.add(w[:-1] + "er")
        if w.endswith("y"):
            inflected.add(w[:-1] + "ies")   # country→countries
            inflected.add(w[:-1] + "ied")   # carry→carried

    _ENGLISH_WORDS = base | inflected

    # Contraction fragments that survive tokenisation
    _ENGLISH_WORDS |= {
        "ve", "ll", "re", "doesn", "hasn", "isn", "wasn", "aren",
        "didn", "won", "wouldn", "shouldn", "couldn", "hadn", "don",
    }

    # Common web/social abbreviations
    _ENGLISH_WORDS |= {
        "http", "https", "www", "url", "html", "jpg", "png", "gif",
        "pdf", "csv", "json", "api", "id", "ok", "lol", "omg", "wtf",
        "idk", "imo", "imho", "fyi", "btw", "tbh", "smh", "ngl",
        "dm", "pm", "rt", "irl", "fwiw", "tldr", "afaik", "etc",
        "covid", "covid19", "ai", "ml", "nft", "crypto", "usa",
    }

    logger.info("English lexicon loaded: %d words (with inflections)", len(_ENGLISH_WORDS))
    return _ENGLISH_WORDS


def _get_evidence(term: str, raw_texts: list[str], doc_indices: list[int], max_n: int = 5) -> list[str]:
    pattern = re.compile(re.escape(term), re.IGNORECASE)
    excerpts: list[str] = []
    for idx in doc_indices:
        if idx < len(raw_texts):
            text = raw_texts[idx]
            if pattern.search(text):
                snippet = text[:300].strip()
                if len(text) > 300:
                    snippet += "..."
                excerpts.append(snippet)
                if len(excerpts) >= max_n:
                    break
    return excerpts


def _is_non_english(term: str, english: set[str]) -> bool:
    """Return True if *term* (possibly multi-word) has at least one non-English word."""
    words = term.split()
    return any(w not in english and len(w) >= 2 for w in words)


def _find_productive_affixes(
    non_english_terms: list[str],
    min_prefix_len: int = 3,
    min_family_size: int = 3,
) -> dict[str, list[str]]:
    """Find prefixes that generate multiple non-English terms.

    Returns a dict mapping each productive prefix to its term family.
    E.g. {"goy": ["goyim", "goyslop", "goybar", "goyfood"]}
    """
    # Only consider single-word terms for affix analysis
    single_words = [t for t in non_english_terms if " " not in t and len(t) >= min_prefix_len + 1]

    prefix_groups: dict[str, list[str]] = defaultdict(list)
    for word in single_words:
        for plen in range(min_prefix_len, min(len(word), len(word) - 1) + 1):
            prefix = word[:plen]
            prefix_groups[prefix].append(word)

    # Keep only prefixes with enough family members
    productive: dict[str, list[str]] = {}
    for prefix, family in sorted(prefix_groups.items(), key=lambda x: len(x[1]), reverse=True):
        unique_family = list(dict.fromkeys(family))
        if len(unique_family) >= min_family_size:
            # Avoid prefixes that are just common English prefixes
            if prefix not in {"the", "pre", "pro", "con", "dis", "mis", "non",
                              "over", "out", "under", "sub", "super", "inter",
                              "trans", "anti", "auto", "semi", "multi", "post",
                              "com", "for", "per", "man", "pol", "can"}:
                productive[prefix] = unique_family

    return productive


class CorpusSpecificDiscovery(DiscoveryModule):
    """Discover terms distinctive to this corpus vs. general English.

    Surfaces neologisms, jargon, loanwords, and productive morphological
    patterns that signal in-group or domain-specific vocabulary.
    """

    def discover(
        self,
        seed_keywords: list[str],
        corpus: Corpus,
        top_n: int = 50,
        min_df: int = 5,
    ) -> list[Candidate]:
        """Find corpus-specific vocabulary not found in standard English."""
        english = _load_english_words()
        seed_set = {s.lower() for s in seed_keywords}

        # Get seed document mask for co-occurrence boost
        seed_mask = get_seed_doc_mask(seed_keywords, corpus)
        n_seed = int(seed_mask.sum())
        n_total = len(corpus.clean_texts)

        # Score non-English terms
        scored: list[tuple[str, float, int]] = []
        non_english_terms: list[str] = []

        for term in corpus.vocab:
            if term in seed_set:
                continue
            df = corpus.doc_freq.get(term, 0)
            if df < min_df:
                continue

            if not _is_non_english(term, english):
                continue

            non_english_terms.append(term)

            # Base score: log(doc_freq) — more frequent non-English terms
            # are more likely to be established jargon vs. typos
            base_score = np.log1p(df)

            # Seed co-occurrence boost: non-English terms that co-occur
            # with seeds are especially interesting
            doc_list = corpus.term_doc_index.get(term, [])
            n_co_seed = sum(1 for d in doc_list if seed_mask[d]) if n_seed > 0 else 0
            seed_ratio = n_co_seed / max(df, 1)

            # Combined score: frequency * (1 + seed_boost)
            score = base_score * (1.0 + 2.0 * seed_ratio)

            scored.append((term, score, df))

        if not scored:
            return []

        # Normalize scores to [0, 1]
        max_score = max(s for _, s, _ in scored)
        if max_score <= 0:
            return []

        scored.sort(key=lambda x: x[1], reverse=True)

        # Find productive affixes
        affixes = _find_productive_affixes(non_english_terms, min_prefix_len=3, min_family_size=3)
        affix_terms: dict[str, str] = {}  # term → prefix
        for prefix, family in affixes.items():
            for t in family:
                if t not in affix_terms:
                    affix_terms[t] = prefix

        candidates: list[Candidate] = []
        for term, raw_score, df in scored[:top_n]:
            norm_score = min(1.0, raw_score / max_score)
            doc_indices = corpus.term_doc_index.get(term, [])
            evidence = _get_evidence(term, corpus.raw_texts, doc_indices)

            # Build context string
            ctx_parts: list[str] = ["Corpus-specific term (not in standard English dictionary)"]
            if term in affix_terms:
                prefix = affix_terms[term]
                family = affixes[prefix]
                ctx_parts.append(
                    f"Productive affix '{prefix}-': {', '.join(family[:5])}"
                )

            candidates.append(
                Candidate(
                    term=term,
                    score=round(norm_score, 4),
                    source="corpus_specific",
                    sources=["corpus_specific"],
                    context=". ".join(ctx_parts),
                    evidence=evidence,
                    doc_count=df,
                )
            )

        return candidates
