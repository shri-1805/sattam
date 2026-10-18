import unittest

from human_rights_graphrag.db import FalkorGraph


class IndexMetadataTests(unittest.TestCase):
    def test_recognizes_a_vector_index_for_article_embeddings(self) -> None:
        row = ["Article", ["embedding"], ["VECTOR"], {"dimension": 384}]

        self.assertTrue(FalkorGraph._is_article_vector_index(row))

    def test_does_not_treat_an_article_range_index_as_vector(self) -> None:
        row = ["Article", ["id"], ["RANGE"], {}]

        self.assertFalse(FalkorGraph._is_article_vector_index(row))

    def test_matches_each_property_index_by_label_property_and_type(self) -> None:
        row = ["SubClause", ["id"], ["RANGE"], {}]

        self.assertTrue(FalkorGraph._has_index(row, "SubClause", "id", "RANGE"))
        self.assertFalse(FalkorGraph._has_index(row, "Article", "id", "RANGE"))


if __name__ == "__main__":
    unittest.main()
