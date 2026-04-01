"""Abstract base class for all keyword discovery modules."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..corpus import Corpus
from ..models import Candidate


class DiscoveryModule(ABC):
    """Base interface that every discovery module must implement."""

    @abstractmethod
    def discover(
        self,
        seed_keywords: list[str],
        corpus: Corpus,
        top_n: int = 50,
        min_df: int = 5,
    ) -> list[Candidate]:
        """Return ranked candidate terms.

        Args:
            seed_keywords: Current seed keyword list.
            corpus: The loaded corpus.
            top_n: Maximum candidates to return.
            min_df: Minimum document frequency threshold.

        Returns:
            Sorted list of :class:`Candidate` objects, highest score first.
        """
        ...
