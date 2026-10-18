"""FalkorDB connection, graph schema, and persistence operations."""

from collections.abc import Sequence
from typing import Any

from redis.exceptions import RedisError

from human_rights_graphrag.models import ArticleContext, SubClauseContext


class DatabaseUnavailableError(RuntimeError):
    """Raised when FalkorDB cannot be reached or a graph query fails."""


class FalkorGraph:
    """Small typed wrapper around the synchronous FalkorDB client."""

    def __init__(
        self,
        host: str,
        port: int,
        graph_name: str,
        password: str | None = None,
    ) -> None:
        try:
            from falkordb import FalkorDB
        except ImportError as exc:
            raise RuntimeError(
                "The FalkorDB Python client is required; install the project requirements"
            ) from exc
        self._client = FalkorDB(host=host, port=port, password=password)
        self._graph = self._client.select_graph(graph_name)

    def _query(
        self, cypher: str, params: dict[str, Any] | None = None
    ) -> list[list[Any]]:
        try:
            result = self._graph.query(cypher, params or {})
            return result.result_set
        except RedisError as exc:
            raise DatabaseUnavailableError(f"FalkorDB query failed: {exc}") from exc

    def health_check(self) -> bool:
        """Verify that the configured graph can execute a trivial query."""
        rows = self._query("RETURN 1 AS healthy")
        return bool(rows and rows[0] and rows[0][0] == 1)

    def ensure_schema(self, embedding_dimension: int) -> None:
        """Create graph property indexes and the article vector index if absent."""
        indexes = self._query("CALL db.indexes()")
        for label, property_name in (
            ("Document", "name"),
            ("Article", "id"),
            ("SubClause", "id"),
            ("Right", "category"),
        ):
            if not any(
                self._has_index(row, label, property_name, "RANGE")
                for row in indexes
            ):
                self._query(
                    f"CREATE INDEX FOR (n:{label}) ON (n.{property_name})"
                )
        if not any(self._is_article_vector_index(row) for row in indexes):
            self._query(
                "CREATE VECTOR INDEX FOR (a:Article) ON (a.embedding) "
                f"OPTIONS {{dimension:{embedding_dimension}, similarityFunction:'cosine'}}"
            )

    @staticmethod
    def _has_index(
        row: Sequence[Any], label: str, property_name: str, index_type: str
    ) -> bool:
        return (
            len(row) >= 3
            and str(row[0]).casefold() == label.casefold()
            and FalkorGraph._metadata_contains(row[1], property_name)
            and FalkorGraph._metadata_contains(row[2], index_type)
        )

    @staticmethod
    def _metadata_contains(value: Any, expected: str) -> bool:
        if isinstance(value, (list, tuple, set)):
            return any(str(item).casefold() == expected.casefold() for item in value)
        return expected.casefold() in str(value).casefold()

    @staticmethod
    def _is_article_vector_index(row: Sequence[Any]) -> bool:
        """Recognize Article.embedding in FalkorDB's index metadata row."""
        return FalkorGraph._has_index(
            row, "Article", "embedding", "VECTOR"
        )

    def upsert_article(
        self,
        *,
        document: str,
        article_id: str,
        reference: str,
        title: str,
        text: str,
        embedding: Sequence[float],
        subclauses: Sequence[SubClauseContext],
        rights: Sequence[str],
    ) -> None:
        """Create or update an article and replace its graph relationships."""
        vector = ", ".join(f"{value:.8f}" for value in embedding)
        self._query(
            "MERGE (d:Document {name: $document}) "
            "MERGE (a:Article {id: $article_id}) "
            "SET a.reference = $reference, a.title = $title, a.text = $text, "
            f"a.embedding = vecf32([{vector}]), a.document = $document "
            "MERGE (d)-[:HAS_ARTICLE]->(a)",
            {
                "document": document,
                "article_id": article_id,
                "reference": reference,
                "title": title,
                "text": text,
            },
        )
        self._query(
            "MATCH (a:Article {id: $article_id}) "
            "MATCH (a)-[old_sub:HAS_SUBCLAUSE]->(:SubClause) "
            "DELETE old_sub",
            {"article_id": article_id},
        )
        self._query(
            "MATCH (a:Article {id: $article_id}) "
            "MATCH (a)-[old_right:PROTECTS]->(:Right) "
            "DELETE old_right",
            {"article_id": article_id},
        )
        for subclause in subclauses:
            self._query(
                "MATCH (a:Article {id: $article_id}) "
                "MERGE (s:SubClause {id: $subclause_id}) "
                "SET s.text = $text "
                "MERGE (a)-[:HAS_SUBCLAUSE]->(s)",
                {
                    "article_id": article_id,
                    "subclause_id": subclause.id,
                    "text": subclause.text,
                },
            )
        for category in rights:
            self._query(
                "MATCH (a:Article {id: $article_id}) "
                "MERGE (r:Right {category: $category}) "
                "MERGE (a)-[:PROTECTS]->(r)",
                {"article_id": article_id, "category": category},
            )

    def search_articles(
        self, embedding: Sequence[float], limit: int
    ) -> list[ArticleContext]:
        """Find semantically similar articles and expand connected provisions."""
        vector = ", ".join(f"{value:.8f}" for value in embedding)
        rows = self._query(
            "CALL db.idx.vector.queryNodes("
            f"'Article', 'embedding', {limit}, vecf32([{vector}])) "
            "YIELD node, score "
            "OPTIONAL MATCH (node)-[:HAS_SUBCLAUSE]->(sub:SubClause) "
            "OPTIONAL MATCH (node)-[:PROTECTS]->(right:Right) "
            "RETURN node.id, node.document, node.reference, node.title, "
            "node.text, score, "
            "collect(DISTINCT {id: sub.id, text: sub.text}), "
            "collect(DISTINCT right.category)"
        )
        return [self._to_context(row) for row in rows]

    @staticmethod
    def _to_context(row: Sequence[Any]) -> ArticleContext:
        subclauses = [
            SubClauseContext(id=item["id"], text=item["text"])
            for item in (row[6] or [])
            if item and item.get("id") and item.get("text")
        ]
        return ArticleContext(
            id=str(row[0]),
            document=str(row[1]),
            reference=str(row[2]),
            title=str(row[3]),
            text=str(row[4]),
            score=float(row[5]),
            subclauses=subclauses,
            rights=[str(value) for value in (row[7] or []) if value],
        )
