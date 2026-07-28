"""Compatibility exports for integrations that imported the original tools module."""

from elite_research.retrieval import (
    DocumentFetcher,
    EvidenceRetriever,
    GoogleSearchProvider,
    WikipediaSearchProvider,
)

__all__ = [
    "DocumentFetcher",
    "EvidenceRetriever",
    "GoogleSearchProvider",
    "WikipediaSearchProvider",
]
