import sys
import unittest
from types import ModuleType
from unittest.mock import MagicMock, patch

from human_rights_graphrag.embeddings import LocalEmbedder


class EmbeddingTokenTests(unittest.TestCase):
    def test_passes_hugging_face_token_to_model_loader(self) -> None:
        model = MagicMock()
        sentence_transformers = ModuleType("sentence_transformers")
        sentence_transformers.SentenceTransformer = MagicMock(return_value=model)

        with patch.dict(sys.modules, {"sentence_transformers": sentence_transformers}):
            embedder = LocalEmbedder("test-model", token="test-hf-token")
            loaded_model = embedder._get_model()

        sentence_transformers.SentenceTransformer.assert_called_once_with(
            "test-model", device="cpu", token="test-hf-token"
        )
        self.assertIs(loaded_model, model)

    def test_omits_token_when_not_configured(self) -> None:
        model = MagicMock()
        sentence_transformers = ModuleType("sentence_transformers")
        sentence_transformers.SentenceTransformer = MagicMock(return_value=model)

        with patch.dict(sys.modules, {"sentence_transformers": sentence_transformers}):
            LocalEmbedder("test-model")._get_model()

        sentence_transformers.SentenceTransformer.assert_called_once_with(
            "test-model", device="cpu"
        )


if __name__ == "__main__":
    unittest.main()
