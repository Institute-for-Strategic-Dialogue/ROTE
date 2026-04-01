"""Pydantic data models for keyword discovery chain state."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Candidate(BaseModel):
    """A single candidate term surfaced by one or more discovery modules."""

    term: str
    score: float = Field(ge=0.0, le=1.0)
    source: str  # pipe-joined when multi-source, e.g. "cooccurrence|network"
    sources: list[str] = Field(default_factory=list)
    context: str = ""
    evidence: list[str] = Field(default_factory=list)
    community_id: int | None = None
    doc_count: int = 0


class ChainRound(BaseModel):
    """One round in the iterative discovery chain."""

    round_number: int
    seed_keywords: list[str]
    added_terms: list[str] = Field(default_factory=list)
    excluded_terms: list[str] = Field(default_factory=list)
    dismissed_terms: list[str] = Field(default_factory=list)
    candidates: list[Candidate] = Field(default_factory=list)


class ChainState(BaseModel):
    """Full state of the keyword discovery chain, persisted in the session."""

    corpus_filename: str = ""
    text_column: str = ""
    total_docs: int = 0
    unique_terms: int = 0
    rounds: list[ChainRound] = Field(default_factory=list)
    current_seeds: list[str] = Field(default_factory=list)
    excluded: list[str] = Field(default_factory=list)
    dismissed: list[str] = Field(default_factory=list)
