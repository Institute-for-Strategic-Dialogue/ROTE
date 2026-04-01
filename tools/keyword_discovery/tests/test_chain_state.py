"""Tests for ChainState serialisation round-trip."""

import pytest

from tools.keyword_discovery.models import Candidate, ChainRound, ChainState


class TestChainStateSerialization:
    def test_round_trip(self):
        state = ChainState(
            corpus_filename="test.csv",
            text_column="text",
            total_docs=100,
            unique_terms=500,
            current_seeds=["globalist", "cabal"],
            dismissed=["the", "and"],
            rounds=[
                ChainRound(
                    round_number=1,
                    seed_keywords=["globalist"],
                    added_terms=["cabal"],
                    dismissed_terms=["the"],
                    candidates=[
                        Candidate(
                            term="cabal",
                            score=0.85,
                            source="cooccurrence",
                            sources=["cooccurrence"],
                            context="Co-occurring term",
                            evidence=["The cabal controls..."],
                            doc_count=5,
                        ),
                        Candidate(
                            term="elite",
                            score=0.72,
                            source="semantic|network",
                            sources=["semantic", "network"],
                            context="Semantically similar",
                            evidence=["Global elites are..."],
                            community_id=2,
                            doc_count=8,
                        ),
                    ],
                )
            ],
        )

        json_str = state.model_dump_json()
        restored = ChainState.model_validate_json(json_str)

        assert restored.corpus_filename == "test.csv"
        assert restored.total_docs == 100
        assert restored.current_seeds == ["globalist", "cabal"]
        assert restored.dismissed == ["the", "and"]
        assert len(restored.rounds) == 1

        rnd = restored.rounds[0]
        assert rnd.round_number == 1
        assert rnd.seed_keywords == ["globalist"]
        assert rnd.added_terms == ["cabal"]
        assert len(rnd.candidates) == 2

        c1 = rnd.candidates[0]
        assert c1.term == "cabal"
        assert c1.score == 0.85
        assert c1.sources == ["cooccurrence"]
        assert c1.evidence == ["The cabal controls..."]

        c2 = rnd.candidates[1]
        assert c2.community_id == 2
        assert c2.sources == ["semantic", "network"]

    def test_empty_state(self):
        state = ChainState()
        json_str = state.model_dump_json()
        restored = ChainState.model_validate_json(json_str)
        assert restored.rounds == []
        assert restored.current_seeds == []
        assert restored.corpus_filename == ""
