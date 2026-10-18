"""Load fundamental-rights provisions into FalkorDB."""

import json
import re
from pathlib import Path
from typing import Any

from human_rights_graphrag.db import FalkorGraph
from human_rights_graphrag.embeddings import LocalEmbedder
from human_rights_graphrag.models import SubClauseContext

DATA_PATH = Path(__file__).parent / "data" / "international_rights.json"
INDIAN_FUNDAMENTAL_RIGHTS_ARTICLES = (
    *(str(number) for number in range(12, 36)),
    "21A",
    "31A",
    "31B",
    "31C",
)
_CLAUSE_MARKER = re.compile(r"(?<!\S)(\(\d+\)|\([a-z]\))")
_RIGHT_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("equality", ("equality", "discrimination", "equal protection")),
    ("life and personal liberty", ("life", "liberty", "personal liberty")),
    (
        "freedom of expression",
        ("expression", "speech", "press", "opinion"),
    ),
    ("freedom of religion", ("religion", "conscience", "worship")),
    (
        "protection from exploitation",
        ("trafficking", "forced labour", "child labour", "exploitation"),
    ),
    (
        "protection from arbitrary detention",
        ("detained", "arrest", "detention", "custody"),
    ),
    ("privacy", ("privacy", "correspondence", "home")),
    ("education", ("education", "educational")),
    (
        "freedom of assembly and association",
        ("assembly", "association", "union"),
    ),
    (
        "freedom of movement and occupation",
        ("movement", "reside", "profession", "occupation", "trade"),
    ),
    (
        "cultural and educational rights",
        ("culture", "language", "minority", "educational institution"),
    ),
    ("right to work", ("work", "employment", "wages")),
    ("constitutional remedies", ("constitutional remedies", "writ")),
)


def split_subclauses(article_id: str, text: str) -> list[SubClauseContext]:
    """Split clearly numbered paragraphs while preserving their source text."""
    matches = list(_CLAUSE_MARKER.finditer(text))
    if not matches:
        return [SubClauseContext(id=f"{article_id}-1", text=text.strip())]
    clauses: list[SubClauseContext] = []
    if text[: matches[0].start()].strip():
        clauses.append(
            SubClauseContext(
                id=f"{article_id}-preamble",
                text=text[: matches[0].start()].strip(),
            )
        )
    marker_occurrences: dict[str, int] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        clause_text = text[match.start() : end].strip()
        if clause_text:
            marker = match.group(1).strip("()")
            marker_occurrences[marker] = marker_occurrences.get(marker, 0) + 1
            occurrence = marker_occurrences[marker]
            suffix = f"-{occurrence}" if occurrence > 1 else ""
            clauses.append(
                SubClauseContext(
                    id=f"{article_id}-{marker}{suffix}", text=clause_text
                )
            )
    return clauses


def classify_rights(title: str, text: str) -> list[str]:
    """Assign only categories supported by explicit terminology in the text."""
    content = f"{title} {text}".casefold()
    return [
        category
        for category, terms in _RIGHT_TERMS
        if any(
            re.search(rf"(?<!\w){re.escape(term)}(?!\w)", content)
            for term in terms
        )
    ]


def _constitution_articles() -> list[tuple[str, str, str]]:
    try:
        from indianconstitution import get_article
    except ImportError as exc:
        raise RuntimeError(
            "indianconstitution is required to ingest Indian constitutional rights"
        ) from exc

    articles: list[tuple[str, str, str]] = []
    for number in INDIAN_FUNDAMENTAL_RIGHTS_ARTICLES:
        article = get_article(number)
        if article is None:
            continue
        title = str(article.title).strip()
        text: Any = article.content
        if not title or not isinstance(text, str) or not text.strip():
            raise ValueError(f"indianconstitution returned incomplete Article {number}")
        articles.append((number, title, text.strip()))
    return articles


def _international_articles() -> list[dict[str, str]]:
    with DATA_PATH.open(encoding="utf-8") as source:
        data = json.load(source)
    return data


def ingest_corpus(graph: FalkorGraph, embedder: LocalEmbedder) -> int:
    """Ingest selected Indian and international fundamental-rights clauses."""
    corpus: list[tuple[str, str, str, str]] = [
        ("Constitution of India", f"Article {number}", title, text)
        for number, title, text in _constitution_articles()
    ]
    corpus.extend(
        (
            item["document"],
            item["id"],
            item["title"],
            item["text"],
        )
        for item in _international_articles()
    )
    if not corpus:
        raise RuntimeError("No legal provisions were loaded for ingestion")

    graph.ensure_schema(embedder.dimension)
    vectors = embedder.encode(
        [f"{title}. {text}" for _, _, title, text in corpus]
    )
    if len(vectors) != len(corpus):
        raise RuntimeError("Embedding model returned an unexpected number of vectors")

    for (document, reference, title, text), embedding in zip(corpus, vectors):
        article_id = (
            f"IND-{reference.removeprefix('Article ')}"
            if document == "Constitution of India"
            else f"{document.upper().replace(' ', '-')}-{reference}"
        )
        graph.upsert_article(
            document=document,
            article_id=article_id,
            reference=reference,
            title=title,
            text=text,
            embedding=embedding,
            subclauses=split_subclauses(article_id, text),
            rights=classify_rights(title, text),
        )
    return len(corpus)
