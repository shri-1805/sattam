"""Scenario concept extraction and graph-backed clause retrieval."""

import re

from human_rights_graphrag.db import FalkorGraph
from human_rights_graphrag.embeddings import LocalEmbedder
from human_rights_graphrag.models import ArticleContext

_STOP_WORDS = frozenset(
    "a about after again all am an and any are as at be because been before "
    "being between both but by can did do does doing down during each few for "
    "from further had has have having he her here hers herself him himself his "
    "how i if in into is it its itself me more most my myself no nor not of off "
    "on once only or other our ours ourselves out over own same she should so "
    "some such than that the their theirs them themselves then there these they "
    "this those through to too under until up very was we were what when where "
    "which while who whom why will with you your yours".split()
)


def extract_concepts(scenario: str, limit: int = 16) -> list[str]:
    """Return the most frequent non-stopword terms from a scenario."""
    words = re.findall(r"[a-zA-Z][a-zA-Z'-]{2,}", scenario.casefold())
    counts: dict[str, int] = {}
    for word in words:
        if word not in _STOP_WORDS:
            counts[word] = counts.get(word, 0) + 1
    return sorted(counts, key=lambda word: (-counts[word], word))[:limit]


class ClauseRetriever:
    """Retrieve article candidates using embeddings and graph traversal."""

    def __init__(
        self, graph: FalkorGraph, embedder: LocalEmbedder, limit: int
    ) -> None:
        self.graph = graph
        self.embedder = embedder
        self.limit = limit

    def retrieve(self, scenario: str) -> tuple[list[str], list[ArticleContext]]:
        concepts = extract_concepts(scenario)
        vectors = self.embedder.encode([scenario])
        if not vectors:
            return concepts, []
        return concepts, self.graph.search_articles(vectors[0], self.limit)
