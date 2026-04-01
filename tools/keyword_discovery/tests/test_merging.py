"""Tests for candidate merging, deduplication, and stem-overlap filtering."""

import pytest

from tools.keyword_discovery.models import Candidate
from tools.keyword_discovery.routes import _merge_candidates, _is_stem_overlap


class TestStemOverlap:
    def test_plural(self):
        assert _is_stem_overlap("rothschilds", {"rothschild"})

    def test_plural_with_article(self):
        assert _is_stem_overlap("the rothschilds", {"rothschild"})

    def test_suffix_variant(self):
        assert _is_stem_overlap("globalists", {"globalist"})

    def test_candidate_is_prefix_of_seed(self):
        # seed is longer than candidate
        assert _is_stem_overlap("global", {"globalist"})

    def test_multi_word_overlap(self):
        assert _is_stem_overlap("great replacements", {"great replacement"})

    def test_no_overlap(self):
        assert not _is_stem_overlap("cabal", {"rothschild"})

    def test_no_overlap_partial_word(self):
        # "rot" is a prefix of "rothschild" but "rotation" shouldn't match
        # Actually "rot" does start "rothschild", so _is_stem_overlap would catch it.
        # This is by design — short seeds cast a wider net.
        # Test a genuinely unrelated term:
        assert not _is_stem_overlap("banking", {"rothschild"})

    def test_exact_match_caught_by_seed_set(self):
        # Exact matches are filtered by seed_set before stem check,
        # but stem_overlap also catches them
        assert _is_stem_overlap("globalist", {"globalist"})

    def test_bigram_candidate_single_seed(self):
        # "new world" contains "new" which is a prefix of seed "new"? No —
        # single-word seed "world" shouldn't match "new world" unless a word matches
        assert _is_stem_overlap("new world", {"world"})
        assert not _is_stem_overlap("new world", {"order"})


class TestMergeCandidates:
    def test_deduplicates_same_term(self):
        c1 = Candidate(
            term="conspiracy", score=0.8, source="cooccurrence",
            sources=["cooccurrence"], context="ctx1", evidence=["e1"]
        )
        c2 = Candidate(
            term="conspiracy", score=0.6, source="semantic",
            sources=["semantic"], context="ctx2", evidence=["e2"]
        )
        merged = _merge_candidates([c1, c2], set(), set(), set(), min_df=0)
        assert len(merged) == 1
        assert merged[0].term == "conspiracy"
        assert merged[0].score == 0.8
        assert set(merged[0].sources) == {"cooccurrence", "semantic"}
        assert "e1" in merged[0].evidence
        assert "e2" in merged[0].evidence

    def test_filters_seeds(self):
        c = Candidate(term="seed_term", score=0.9, source="cooccurrence", sources=["cooccurrence"])
        merged = _merge_candidates([c], {"seed_term"}, set(), set(), min_df=0)
        assert len(merged) == 0

    def test_filters_dismissed(self):
        c = Candidate(term="bad_term", score=0.9, source="cooccurrence", sources=["cooccurrence"])
        merged = _merge_candidates([c], set(), {"bad_term"}, set(), min_df=0)
        assert len(merged) == 0

    def test_filters_stem_overlap(self):
        c = Candidate(term="rothschilds", score=0.9, source="cooccurrence", sources=["cooccurrence"])
        merged = _merge_candidates([c], {"rothschild"}, set(), set(), min_df=0)
        assert len(merged) == 0

    def test_filters_stem_overlap_with_article(self):
        c = Candidate(term="the rothschilds", score=0.9, source="cooccurrence", sources=["cooccurrence"])
        merged = _merge_candidates([c], {"rothschild"}, set(), set(), min_df=0)
        assert len(merged) == 0

    def test_filters_excluded_exact(self):
        c = Candidate(term="reporter", score=0.9, source="cooccurrence", sources=["cooccurrence"])
        merged = _merge_candidates([c], set(), set(), {"reporter"}, min_df=0)
        assert len(merged) == 0

    def test_filters_excluded_stem(self):
        c = Candidate(term="reporters", score=0.9, source="cooccurrence", sources=["cooccurrence"])
        merged = _merge_candidates([c], set(), set(), {"reporter"}, min_df=0)
        assert len(merged) == 0

    def test_filters_single_char_terms(self):
        c = Candidate(term="x", score=0.9, source="cooccurrence", sources=["cooccurrence"])
        merged = _merge_candidates([c], set(), set(), set(), min_df=0)
        assert len(merged) == 0

    def test_allows_two_char_terms(self):
        c = Candidate(term="hh", score=0.9, source="cooccurrence", sources=["cooccurrence"])
        merged = _merge_candidates([c], set(), set(), set(), min_df=0)
        assert len(merged) == 1

    def test_filters_stopwords(self):
        c = Candidate(term="the", score=0.9, source="cooccurrence", sources=["cooccurrence"])
        merged = _merge_candidates([c], set(), set(), set(), min_df=0)
        assert len(merged) == 0

    def test_sorted_by_score(self):
        candidates = [
            Candidate(term="low", score=0.3, source="cooccurrence", sources=["cooccurrence"]),
            Candidate(term="high", score=0.9, source="semantic", sources=["semantic"]),
            Candidate(term="mid", score=0.6, source="network", sources=["network"]),
        ]
        merged = _merge_candidates(candidates, set(), set(), set(), min_df=0)
        scores = [c.score for c in merged]
        assert scores == sorted(scores, reverse=True)

    def test_multi_source_context(self):
        c1 = Candidate(
            term="bridge", score=0.7, source="cooccurrence",
            sources=["cooccurrence"], context="Co-occurring"
        )
        c2 = Candidate(
            term="bridge", score=0.9, source="network",
            sources=["network"], context="Graph bridge", community_id=3
        )
        merged = _merge_candidates([c1, c2], set(), set(), set(), min_df=0)
        assert len(merged) == 1
        assert set(merged[0].source.split("|")) == {"network", "cooccurrence"}
        assert merged[0].community_id == 3
