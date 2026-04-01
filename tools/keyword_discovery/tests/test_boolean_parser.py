"""Tests for the boolean expression parser and corpus filter."""

import numpy as np
import pytest

from tools.keyword_discovery.corpus import (
    Corpus,
    apply_boolean_filter,
    _looks_boolean,
    _build_boolean_parser,
)


class TestLooksBoolean:
    def test_simple_keywords(self):
        assert not _looks_boolean("globalist, cabal, soros")

    def test_and(self):
        assert _looks_boolean("globalist AND cabal")

    def test_or(self):
        assert _looks_boolean("globalist OR cabal")

    def test_not(self):
        assert _looks_boolean("globalist NOT news")

    def test_parens(self):
        assert _looks_boolean("(globalist OR cabal)")

    def test_case_insensitive(self):
        assert _looks_boolean("globalist and cabal")


class TestBooleanParser:
    def test_single_term(self, small_corpus):
        mask = apply_boolean_filter("globalist", small_corpus)
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert mask.sum() > 0

    def test_or_expression(self, small_corpus):
        mask_a = apply_boolean_filter("globalist", small_corpus)
        mask_b = apply_boolean_filter("cat", small_corpus)
        mask_or = apply_boolean_filter("globalist OR cat", small_corpus)
        # OR should match at least as many docs as either alone
        assert mask_or.sum() >= mask_a.sum()
        assert mask_or.sum() >= mask_b.sum()

    def test_and_expression(self, small_corpus):
        mask_and = apply_boolean_filter("globalist AND cabal", small_corpus)
        mask_a = apply_boolean_filter("globalist", small_corpus)
        # AND should match fewer or equal docs
        assert mask_and.sum() <= mask_a.sum()

    def test_not_expression(self, small_corpus):
        mask_all = apply_boolean_filter("globalist", small_corpus)
        mask_not = apply_boolean_filter("globalist AND NOT cabal", small_corpus)
        assert mask_not.sum() <= mask_all.sum()

    def test_phrase_matching(self, small_corpus):
        mask = apply_boolean_filter('"great replacement"', small_corpus)
        assert mask.sum() > 0

    def test_wildcard(self, small_corpus):
        mask = apply_boolean_filter("global*", small_corpus)
        assert mask.sum() > 0

    def test_parenthetical_grouping(self, small_corpus):
        mask = apply_boolean_filter("(globalist OR cabal) AND NOT cat", small_corpus)
        assert isinstance(mask, np.ndarray)
        assert mask.sum() > 0

    def test_comma_separated_fallback(self, small_corpus):
        mask = apply_boolean_filter("globalist, cabal, cat", small_corpus)
        assert mask.sum() > 0

    def test_empty_expression(self, small_corpus):
        mask = apply_boolean_filter("", small_corpus)
        # Empty filter should match everything
        assert mask.sum() == len(small_corpus.clean_texts)
