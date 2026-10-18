"""Local sentence embeddings used for article indexing and retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class LocalEmbedder:
    """Lazily load a CPU sentence-transformer model."""

    def __init__(self, model_name: str, token: str | None = None) -> None:
        self.model_name = model_name
        self.token = token
        self._model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for embeddings; "
                    "install the project requirements first"
                ) from exc
            model_options: dict[str, str] = {"device": "cpu"}
            if self.token:
                model_options["token"] = self.token
            self._model = SentenceTransformer(self.model_name, **model_options)
        return self._model

    @property
    def dimension(self) -> int:
        dimension = self._get_model().get_sentence_embedding_dimension()
        if dimension is None:
            raise RuntimeError(
                f"Embedding model {self.model_name!r} did not report its dimension"
            )
        return int(dimension)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._get_model().encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(value) for value in vector] for vector in vectors]
