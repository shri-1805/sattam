import unittest

from human_rights_graphrag.retriever import extract_concepts


class ConceptExtractionTests(unittest.TestCase):
    def test_removes_common_words_and_prioritizes_frequent_terms(self) -> None:
        concepts = extract_concepts(
            "The officer detained the peaceful protester; the protester was detained."
        )

        self.assertIn("detained", concepts)
        self.assertIn("protester", concepts)
        self.assertNotIn("the", concepts)
        self.assertEqual(concepts[:2], ["detained", "protester"])

    def test_limits_number_of_concepts(self) -> None:
        concepts = extract_concepts("alpha bravo charlie delta echo foxtrot", limit=3)

        self.assertEqual(len(concepts), 3)


if __name__ == "__main__":
    unittest.main()
