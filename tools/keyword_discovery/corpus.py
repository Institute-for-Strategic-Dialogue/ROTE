"""Corpus loading, normalisation, term indexing, and boolean search/filter."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from pyparsing import (
    CaselessKeyword,
    Forward,
    Group,
    ParserElement,
    QuotedString,
    Regex,
    infix_notation,
    opAssoc,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+")
# Capitalized word sequences (2-4 words) — likely proper nouns / named entities
_PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b")
_TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)
# Match individual emoji or runs of identical emoji (e.g. ⚡️⚡️ → one token)
_EMOJI_RE = re.compile(
    r"([\U0001F300-\U0001F9FF\u2600-\u27BF\u2700-\u27BF\u200D"
    r"\uFE0F\u2B50\u2764\u270A-\u270D\u2640\u2642\u2695\u2696"
    r"\u2708\u2744\u2B06\u2194-\u21AA\u23CF\u23E9-\u23F3"
    r"\u25AA-\u25FE\u2611\u2660-\u2668\u267B\u267F\u2692-\u269C"
    r"\u26A0-\u26FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF]+)",
    re.UNICODE,
)

# Common text-column names (checked case-insensitively)
_TEXT_COLUMN_NAMES = [
    "text", "content", "body", "message", "full_text", "post_text",
]


# ---------------------------------------------------------------------------
# Corpus data class
# ---------------------------------------------------------------------------
@dataclass
class Corpus:
    """Holds the loaded corpus, its term index, and metadata."""

    df: pd.DataFrame
    text_column: str
    raw_texts: list[str]
    clean_texts: list[str]
    # term -> sorted list of doc indices where it appears
    term_doc_index: dict[str, list[int]] = field(default_factory=dict)
    # ordered vocabulary (terms above min_df)
    vocab: list[str] = field(default_factory=list)
    # doc-frequency for every vocab term
    doc_freq: dict[str, int] = field(default_factory=dict)
    # content hash for caching
    content_hash: str = ""
    ngram_range: tuple[int, int] = (1, 2)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _detect_text_column(df: pd.DataFrame, user_col: str | None) -> str:
    """Return the best text column name from *df*."""
    if user_col and user_col in df.columns:
        return user_col
    lower_map = {c.lower(): c for c in df.columns}
    for name in _TEXT_COLUMN_NAMES:
        if name in lower_map:
            return lower_map[name]
    # Fall back to first string-typed column
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            return col
    return df.columns[0]


def _clean(text: str) -> str:
    """Lowercase, strip URLs and @mentions."""
    text = _URL_RE.sub("", text)
    text = _MENTION_RE.sub("", text)
    return text.lower().strip()


def _tokenize_ngrams(text: str, ngram_range: tuple[int, int] = (1, 2)) -> list[str]:
    """Extract token n-grams plus emoji tokens from *text*."""
    tokens = _TOKEN_RE.findall(text)
    out: list[str] = []
    lo, hi = ngram_range
    for n in range(lo, hi + 1):
        for i in range(len(tokens) - n + 1):
            out.append(" ".join(tokens[i : i + n]))
    # Also extract emoji sequences as standalone tokens
    for m in _EMOJI_RE.finditer(text):
        emoji_tok = m.group(0).strip()
        if emoji_tok:
            out.append(emoji_tok)
    return out


def load_corpus(
    file_bytes: bytes,
    filename: str,
    user_column: str | None = None,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
) -> Corpus:
    """Load a CSV or JSON file into a :class:`Corpus`.

    Args:
        file_bytes: Raw bytes of the uploaded file.
        filename: Original filename (used to detect format).
        user_column: Optional user-specified text column name.
        ngram_range: (min_n, max_n) for n-gram tokenisation.
        min_df: Minimum document frequency for a term to be indexed.

    Returns:
        A populated :class:`Corpus`.
    """
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext == "json":
        df = pd.read_json(file_bytes.decode("utf-8", errors="replace"))
    else:
        # Default CSV
        try:
            df = pd.read_csv(
                pd.io.common.BytesIO(file_bytes),
                encoding="utf-8",
                on_bad_lines="skip",
            )
        except UnicodeDecodeError:
            df = pd.read_csv(
                pd.io.common.BytesIO(file_bytes),
                encoding="latin-1",
                on_bad_lines="skip",
            )

    text_col = _detect_text_column(df, user_column)
    df[text_col] = df[text_col].astype(str).fillna("")

    raw_texts: list[str] = df[text_col].tolist()
    clean_texts: list[str] = [_clean(t) for t in raw_texts]

    # Build term → doc index
    term_doc_index: dict[str, list[int]] = {}
    for doc_idx, (text, raw) in enumerate(zip(clean_texts, raw_texts)):
        seen: set[str] = set()
        for term in _tokenize_ngrams(text, ngram_range):
            if term not in seen:
                seen.add(term)
                term_doc_index.setdefault(term, []).append(doc_idx)
        # Also extract capitalized proper noun sequences from RAW text
        # (2-4 word names like "Julius Streicher", "E Michael Jones")
        # Stored lowercased so they merge with the main index
        for m in _PROPER_NOUN_RE.finditer(raw):
            name = m.group(0).lower()
            if name not in seen:
                seen.add(name)
                term_doc_index.setdefault(name, []).append(doc_idx)

    # Filter to terms above min_df
    doc_freq = {t: len(docs) for t, docs in term_doc_index.items() if len(docs) >= min_df}
    term_doc_index = {t: docs for t, docs in term_doc_index.items() if t in doc_freq}
    vocab = sorted(doc_freq.keys())

    content_hash = hashlib.md5(file_bytes[:65536]).hexdigest()

    logger.info(
        "Corpus loaded: %d docs, %d unique terms (min_df=%d)",
        len(df),
        len(vocab),
        min_df,
    )

    return Corpus(
        df=df,
        text_column=text_col,
        raw_texts=raw_texts,
        clean_texts=clean_texts,
        term_doc_index=term_doc_index,
        vocab=vocab,
        doc_freq=doc_freq,
        content_hash=content_hash,
        ngram_range=ngram_range,
    )


# ---------------------------------------------------------------------------
# Boolean expression search / filter
# ---------------------------------------------------------------------------

def _looks_boolean(expr: str) -> bool:
    """Return True if *expr* contains boolean operators, parens, quotes, or wildcards."""
    upper = expr.upper()
    if any(kw in upper for kw in (" AND ", " OR ", " NOT ", "(", ")")):
        return True
    if '"' in expr:
        return True
    if re.search(r"\w\*", expr):
        return True
    return False


def _build_boolean_parser() -> ParserElement:
    """Build a pyparsing grammar for boolean keyword expressions.

    Supports:
        - AND, OR, NOT (case-insensitive)
        - Parenthetical grouping
        - Quoted phrases: ``"exact phrase"``
        - Wildcard suffix: ``disinform*``
    """
    AND = CaselessKeyword("AND").suppress()
    OR = CaselessKeyword("OR").suppress()
    NOT = CaselessKeyword("NOT")

    phrase = QuotedString('"', unquote_results=True)
    wildcard_term = Regex(r"\b\w+\*")
    plain_term = Regex(r"\b(?!(?:and|or|not)\b)\w+\b", re.IGNORECASE)

    atom = phrase | wildcard_term | plain_term

    expr = infix_notation(
        atom,
        [
            (NOT, 1, opAssoc.RIGHT, lambda t: ("NOT", t[0][1])),
            (AND, 2, opAssoc.LEFT, lambda t: ("AND", list(t[0]))),
            (OR, 2, opAssoc.LEFT, lambda t: ("OR", list(t[0]))),
        ],
    )
    return expr


_BOOL_PARSER = _build_boolean_parser()


def _eval_node(node, texts: Sequence[str]) -> np.ndarray:
    """Recursively evaluate a parsed boolean expression against *texts*.

    Returns a boolean numpy array (mask) of length ``len(texts)``.
    """
    n = len(texts)

    if isinstance(node, str):
        # Leaf node — a plain term, phrase, or wildcard
        if node.endswith("*"):
            prefix = node[:-1].lower()
            pattern = re.compile(r"\b" + re.escape(prefix) + r"\w*", re.IGNORECASE)
            return np.array([bool(pattern.search(t)) for t in texts])
        else:
            term_lower = node.lower()
            if " " in term_lower:
                # Phrase
                return np.array([term_lower in t.lower() for t in texts])
            else:
                pattern = re.compile(r"\b" + re.escape(term_lower) + r"\b", re.IGNORECASE)
                return np.array([bool(pattern.search(t)) for t in texts])

    if isinstance(node, tuple):
        op = node[0]
        if op == "NOT":
            return ~_eval_node(node[1], texts)
        if op == "AND":
            result = np.ones(n, dtype=bool)
            for child in node[1]:
                result &= _eval_node(child, texts)
            return result
        if op == "OR":
            result = np.zeros(n, dtype=bool)
            for child in node[1]:
                result |= _eval_node(child, texts)
            return result

    # pyparsing may wrap in a list
    if isinstance(node, list):
        if len(node) == 1:
            return _eval_node(node[0], texts)
        # Implicit AND
        result = np.ones(n, dtype=bool)
        for child in node:
            result &= _eval_node(child, texts)
        return result

    # Fallback
    return np.ones(n, dtype=bool)


def apply_boolean_filter(expr: str, corpus: Corpus) -> np.ndarray:
    """Parse *expr* and return a boolean mask over the corpus.

    If *expr* doesn't contain boolean operators it is treated as a
    comma-separated keyword list (implicit OR).
    """
    expr = expr.strip()
    if not expr:
        return np.ones(len(corpus.clean_texts), dtype=bool)

    if _looks_boolean(expr):
        try:
            parsed = _BOOL_PARSER.parse_string(expr, parse_all=True)
            return _eval_node(parsed.asList(), corpus.clean_texts)
        except Exception:
            logger.warning("Boolean parse failed for %r, falling back to keyword list", expr)

    # Comma-separated keyword list — implicit OR
    keywords = [kw.strip().lower() for kw in expr.split(",") if kw.strip()]
    mask = np.zeros(len(corpus.clean_texts), dtype=bool)
    for kw in keywords:
        pattern = re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
        for i, text in enumerate(corpus.clean_texts):
            if not mask[i] and pattern.search(text):
                mask[i] = True
    return mask


def get_seed_doc_mask(seeds: list[str], corpus: Corpus) -> np.ndarray:
    """Return a boolean mask of docs containing at least one seed keyword."""
    mask = np.zeros(len(corpus.clean_texts), dtype=bool)
    for seed in seeds:
        seed_lower = seed.lower()
        if seed_lower in corpus.term_doc_index:
            for idx in corpus.term_doc_index[seed_lower]:
                mask[idx] = True
        else:
            # Regex fallback for multi-word seeds or partial matches
            pattern = re.compile(r"\b" + re.escape(seed_lower) + r"\b")
            for i, text in enumerate(corpus.clean_texts):
                if not mask[i] and pattern.search(text):
                    mask[i] = True
    return mask
