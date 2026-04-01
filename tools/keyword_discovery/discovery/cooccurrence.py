"""Co-occurrence discovery via PMI and Log-Likelihood Ratio."""

from __future__ import annotations

import logging
import math
import re
from typing import Sequence

import numpy as np

from ..corpus import Corpus
from ..models import Candidate
from .base import DiscoveryModule

logger = logging.getLogger(__name__)


def _pmi(
    n_xy: int, n_x: int, n_y: int, n_total: int
) -> float:
    """Pointwise Mutual Information (base 2).

    PMI(x,y) = log2(P(x,y) / (P(x) * P(y)))
    """
    if n_xy == 0 or n_x == 0 or n_y == 0 or n_total == 0:
        return 0.0
    p_xy = n_xy / n_total
    p_x = n_x / n_total
    p_y = n_y / n_total
    return math.log2(p_xy / (p_x * p_y))


def _llr(
    n_xy: int, n_x: int, n_y: int, n_total: int
) -> float:
    """Log-likelihood ratio statistic.

    Uses the G^2 formulation for a 2x2 contingency table.
    """
    a = n_xy
    b = n_x - n_xy
    c = n_y - n_xy
    d = n_total - n_x - n_y + n_xy

    def _log_l(k: int, n: int, p: float) -> float:
        if k == 0 or n == 0:
            return 0.0
        p = max(1e-10, min(1 - 1e-10, p))
        return k * math.log(p) + (n - k) * math.log(1 - p)

    total = a + b + c + d
    if total == 0:
        return 0.0

    p1 = (a + b) / total if total else 0.5
    p2 = (a + c) / total if total else 0.5

    row1 = a + b
    row2 = c + d
    p_row1 = a / row1 if row1 else 0.5
    p_row2 = c / row2 if row2 else 0.5

    ll_h0 = _log_l(a, row1, p2) + _log_l(c, row2, p2)
    ll_h1 = _log_l(a, row1, p_row1) + _log_l(c, row2, p_row2)

    return 2.0 * (ll_h1 - ll_h0)


def _minmax_normalize(values: Sequence[float]) -> list[float]:
    """Normalize values to [0, 1]."""
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    rng = hi - lo
    if rng == 0:
        return [0.5] * len(values)
    return [(v - lo) / rng for v in values]


def _get_evidence(term: str, raw_texts: list[str], doc_indices: list[int], max_n: int = 5) -> list[str]:
    """Extract sample excerpts containing *term*."""
    pattern = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
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


class CooccurrenceDiscovery(DiscoveryModule):
    """Discover terms that statistically co-occur with seed keywords.

    Uses a blend of PMI and LLR scores at the document level.
    """

    def __init__(self, pmi_weight: float = 0.4, llr_weight: float = 0.6):
        self.pmi_weight = pmi_weight
        self.llr_weight = llr_weight

    def discover(
        self,
        seed_keywords: list[str],
        corpus: Corpus,
        top_n: int = 50,
        min_df: int = 5,
    ) -> list[Candidate]:
        """Compute PMI/LLR co-occurrence scores for all corpus terms against seeds."""
        n_total = len(corpus.clean_texts)
        if n_total == 0:
            return []

        # Collect doc indices for seed keywords
        seed_set = {s.lower() for s in seed_keywords}
        seed_docs: set[int] = set()
        for seed in seed_set:
            if seed in corpus.term_doc_index:
                seed_docs.update(corpus.term_doc_index[seed])
        n_seed = len(seed_docs)
        if n_seed == 0:
            logger.info("Co-occurrence: no seed docs found")
            return []

        # Precompute seed membership as a boolean array for fast intersection
        seed_mask = np.zeros(n_total, dtype=bool)
        for idx in seed_docs:
            seed_mask[idx] = True

        # Score each vocab term — only iterate terms above min_df
        raw_pmi: dict[str, float] = {}
        raw_llr: dict[str, float] = {}
        term_seed_docs: dict[str, list[int]] = {}

        for term in corpus.vocab:
            if term in seed_set:
                continue
            term_docs_list = corpus.term_doc_index.get(term, [])
            n_term = len(term_docs_list)
            if n_term < min_df:
                continue

            # Count co-occurring docs via the precomputed mask
            n_co = sum(1 for d in term_docs_list if seed_mask[d])
            if n_co == 0:
                continue

            raw_pmi[term] = _pmi(n_co, n_seed, n_term, n_total)
            raw_llr[term] = _llr(n_co, n_seed, n_term, n_total)
            # Keep a small sample of co-occurring doc indices for evidence
            sample = [d for d in term_docs_list if seed_mask[d]]
            term_seed_docs[term] = sample[:20]

        if not raw_pmi:
            return []

        # Normalize and blend
        terms = list(raw_pmi.keys())
        pmi_vals = _minmax_normalize([raw_pmi[t] for t in terms])
        llr_vals = _minmax_normalize([raw_llr[t] for t in terms])

        scored: list[tuple[str, float]] = []
        for i, term in enumerate(terms):
            blended = self.pmi_weight * pmi_vals[i] + self.llr_weight * llr_vals[i]
            scored.append((term, blended))

        scored.sort(key=lambda x: x[1], reverse=True)

        candidates: list[Candidate] = []
        for term, score in scored[:top_n]:
            evidence = _get_evidence(term, corpus.raw_texts, term_seed_docs.get(term, []))
            doc_count = corpus.doc_freq.get(term, 0)
            pmi_val = raw_pmi.get(term, 0)
            llr_val = raw_llr.get(term, 0)
            candidates.append(
                Candidate(
                    term=term,
                    score=round(score, 4),
                    source="cooccurrence",
                    sources=["cooccurrence"],
                    context=f"Co-occurring term (PMI: {pmi_val:.2f}, LLR: {llr_val:.1f})",
                    evidence=evidence,
                    doc_count=doc_count,
                )
            )

        return candidates
