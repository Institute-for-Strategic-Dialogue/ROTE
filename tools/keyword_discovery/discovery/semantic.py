"""Distributional similarity discovery via PPMI + SVD.

Trains corpus-specific word embeddings by building a windowed
co-occurrence matrix, weighting with Positive PMI, and reducing
dimensions with truncated SVD.  Words that appear in the same
contexts as seed keywords in *this* corpus land nearby in the
resulting vector space.
"""

from __future__ import annotations

import logging
import re

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import svds

from ..corpus import Corpus
from ..models import Candidate
from .base import DiscoveryModule

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)


def _get_evidence(term: str, raw_texts: list[str], doc_indices: list[int], max_n: int = 5) -> list[str]:
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


class SemanticDiscovery(DiscoveryModule):
    """Find terms used in similar contexts to seeds via corpus-trained embeddings.

    Builds a PPMI-weighted co-occurrence matrix from the corpus with a
    sliding window, reduces it with truncated SVD, then ranks vocabulary
    terms by cosine similarity to the seed centroid.
    """

    def __init__(
        self,
        window: int = 5,
        svd_dims: int = 100,
        max_vocab: int = 10000,
    ):
        self.window = window
        self.svd_dims = svd_dims
        self.max_vocab = max_vocab

    def discover(
        self,
        seed_keywords: list[str],
        corpus: Corpus,
        top_n: int = 50,
        min_df: int = 5,
    ) -> list[Candidate]:
        """Build corpus embeddings and find nearest neighbours to seeds."""
        seed_set = {s.lower() for s in seed_keywords}

        # Build vocabulary: top terms by doc freq (including seeds)
        sorted_terms = sorted(
            ((t, f) for t, f in corpus.doc_freq.items() if f >= min_df),
            key=lambda x: x[1],
            reverse=True,
        )
        vocab_list = [t for t, _ in sorted_terms[: self.max_vocab]]
        if not vocab_list:
            return []
        vocab_to_idx = {t: i for i, t in enumerate(vocab_list)}
        vocab_set = set(vocab_list)
        n_vocab = len(vocab_list)

        # Check that at least one seed is in vocab
        seed_indices = [vocab_to_idx[s] for s in seed_set if s in vocab_to_idx]
        if not seed_indices:
            logger.info("Distributional: no seed terms found in top-%d vocab", self.max_vocab)
            return []

        # Build windowed co-occurrence matrix (sparse)
        cooc = lil_matrix((n_vocab, n_vocab), dtype=np.float32)
        window = self.window

        for text in corpus.clean_texts:
            tokens = _TOKEN_RE.findall(text)
            # Map to vocab indices, skip unknown tokens
            token_ids = [vocab_to_idx[t] for t in tokens if t in vocab_set]
            for i, tid in enumerate(token_ids):
                lo = max(0, i - window)
                hi = min(len(token_ids), i + window + 1)
                for j in range(lo, hi):
                    if i != j:
                        cooc[tid, token_ids[j]] += 1.0

        cooc = cooc.tocsr()

        # Compute Positive PMI
        total = cooc.sum()
        if total == 0:
            return []

        row_sums = np.array(cooc.sum(axis=1)).flatten()
        col_sums = np.array(cooc.sum(axis=0)).flatten()

        # Avoid division by zero
        row_sums[row_sums == 0] = 1
        col_sums[col_sums == 0] = 1

        # For each non-zero entry: PMI = log(p(x,y) / (p(x)*p(y)))
        #   = log(count(x,y) * total / (row_sum(x) * col_sum(y)))
        cx = cooc.tocoo()
        pmi_vals = np.log(
            (cx.data * total) / (row_sums[cx.row] * col_sums[cx.col])
        )
        # Positive PMI: clamp to zero
        pmi_vals = np.maximum(pmi_vals, 0)
        ppmi = csr_matrix((pmi_vals, (cx.row, cx.col)), shape=(n_vocab, n_vocab))

        # Truncated SVD to get dense embeddings
        k = min(self.svd_dims, n_vocab - 1, ppmi.shape[0] - 1)
        if k < 2:
            return []

        try:
            U, S, _ = svds(ppmi.astype(np.float64), k=k)
        except Exception:
            logger.exception("SVD failed")
            return []

        # Weight by sqrt(S) for better results (Levy & Goldberg 2014)
        embeddings = U * np.sqrt(S)

        # Normalize rows
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms

        # Compute seed centroid
        seed_vecs = embeddings[seed_indices]
        centroid = seed_vecs.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm == 0:
            return []
        centroid = centroid / centroid_norm

        # Cosine similarity to centroid
        sims = embeddings @ centroid

        # Also compute max sim to any individual seed
        if len(seed_indices) > 1:
            all_sims = embeddings @ seed_vecs.T
            max_individual = all_sims.max(axis=1)
            blended = 0.5 * sims + 0.5 * max_individual
        else:
            blended = sims

        # Zero out seeds
        for si in seed_indices:
            blended[si] = -1.0

        # Normalize to [0, 1]
        valid_mask = blended > -0.5
        if not valid_mask.any():
            return []
        lo = float(blended[valid_mask].min())
        hi = float(blended[valid_mask].max())
        rng = hi - lo if hi > lo else 1.0

        ranked_indices = np.argsort(-blended)
        candidates: list[Candidate] = []
        for idx in ranked_indices:
            idx = int(idx)
            if idx in seed_indices:
                continue
            term = vocab_list[idx]
            if term in seed_set:
                continue
            raw_sim = float(blended[idx])
            score = (raw_sim - lo) / rng
            if score <= 0:
                continue

            doc_indices = corpus.term_doc_index.get(term, [])
            evidence = _get_evidence(term, corpus.raw_texts, doc_indices)
            doc_count = corpus.doc_freq.get(term, 0)

            # Find closest seed
            best_seed = seed_keywords[0]
            best_sim = raw_sim
            if len(seed_indices) > 1:
                individual_sims = embeddings[idx] @ seed_vecs.T
                best_si = int(individual_sims.argmax())
                seed_list = [s for s in seed_set if s in vocab_to_idx]
                if best_si < len(seed_list):
                    best_seed = seed_list[best_si]
                    best_sim = float(individual_sims[best_si])

            candidates.append(
                Candidate(
                    term=term,
                    score=round(min(1.0, score), 4),
                    source="semantic",
                    sources=["semantic"],
                    context=f"Used in similar contexts to '{best_seed}' (distributional sim: {best_sim:.3f})",
                    evidence=evidence,
                    doc_count=doc_count,
                )
            )
            if len(candidates) >= top_n:
                break

        return candidates
