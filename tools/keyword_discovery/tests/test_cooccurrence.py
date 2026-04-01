"""Tests for PMI / LLR computation and co-occurrence discovery."""

import math
import pytest

from tools.keyword_discovery.discovery.cooccurrence import (
    _pmi,
    _llr,
    _minmax_normalize,
    CooccurrenceDiscovery,
)


class TestPMI:
    def test_perfect_cooccurrence(self):
        # x and y always appear together in 10 of 100 docs
        pmi = _pmi(n_xy=10, n_x=10, n_y=10, n_total=100)
        # log2(0.1 / (0.1 * 0.1)) = log2(10) ~ 3.32
        assert pmi == pytest.approx(math.log2(10), rel=1e-3)

    def test_independent(self):
        # x appears in 50, y in 50, both in 25 of 100 docs
        pmi = _pmi(n_xy=25, n_x=50, n_y=50, n_total=100)
        assert pmi == pytest.approx(0.0, abs=1e-6)

    def test_zero_cooccurrence(self):
        assert _pmi(0, 10, 10, 100) == 0.0

    def test_zero_total(self):
        assert _pmi(5, 10, 10, 0) == 0.0


class TestLLR:
    def test_strong_association(self):
        llr = _llr(n_xy=50, n_x=60, n_y=55, n_total=100)
        assert llr > 0

    def test_no_association(self):
        # Independent: 25 co-occur out of 50 each in 100
        llr = _llr(n_xy=25, n_x=50, n_y=50, n_total=100)
        assert abs(llr) < 1.0  # Should be near zero

    def test_zero_cooccurrence(self):
        llr = _llr(n_xy=0, n_x=10, n_y=10, n_total=100)
        # LLR should handle gracefully
        assert isinstance(llr, float)


class TestMinMaxNormalize:
    def test_normal(self):
        result = _minmax_normalize([1, 2, 3, 4, 5])
        assert result[0] == 0.0
        assert result[-1] == 1.0

    def test_uniform(self):
        result = _minmax_normalize([3, 3, 3])
        assert all(v == 0.5 for v in result)

    def test_empty(self):
        assert _minmax_normalize([]) == []


class TestCooccurrenceDiscovery:
    def test_discovers_related_terms(self, small_corpus):
        disc = CooccurrenceDiscovery()
        candidates = disc.discover(["globalist"], small_corpus, top_n=10, min_df=1)
        assert len(candidates) > 0
        terms = [c.term for c in candidates]
        # Expect terms that co-occur with "globalist" in the test corpus
        assert all(c.source == "cooccurrence" for c in candidates)
        assert all(0 <= c.score <= 1 for c in candidates)

    def test_no_seeds_in_results(self, small_corpus):
        disc = CooccurrenceDiscovery()
        candidates = disc.discover(["globalist", "cabal"], small_corpus, top_n=10, min_df=1)
        terms = {c.term for c in candidates}
        assert "globalist" not in terms
        assert "cabal" not in terms

    def test_empty_corpus(self):
        from tools.keyword_discovery.corpus import Corpus
        import pandas as pd

        corpus = Corpus(
            df=pd.DataFrame(),
            text_column="text",
            raw_texts=[],
            clean_texts=[],
        )
        disc = CooccurrenceDiscovery()
        candidates = disc.discover(["test"], corpus, top_n=10, min_df=1)
        assert candidates == []
