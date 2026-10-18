import unittest

from human_rights_graphrag.ingest import classify_rights, split_subclauses


class IngestionTests(unittest.TestCase):
    def test_splits_numbered_subclauses_without_changing_their_text(self) -> None:
        text = "(1) Every person has liberty.\n(2) No one shall be detained arbitrarily."

        clauses = split_subclauses("IND-21", text)

        self.assertEqual([clause.id for clause in clauses], ["IND-21-1", "IND-21-2"])
        self.assertEqual(clauses[0].text, "(1) Every person has liberty.")
        self.assertEqual(clauses[1].text, "(2) No one shall be detained arbitrarily.")

    def test_unsplit_article_is_retained_verbatim(self) -> None:
        text = "Everyone has the right to life and liberty."

        clauses = split_subclauses("UDHR-3", text)

        self.assertEqual(len(clauses), 1)
        self.assertEqual(clauses[0].text, text)

    def test_repeated_nested_markers_receive_unique_ids(self) -> None:
        clauses = split_subclauses(
            "IND-19",
            "(1) First paragraph. (a) First item. (2) Second paragraph. (a) Second item.",
        )

        self.assertEqual(
            [clause.id for clause in clauses],
            ["IND-19-1", "IND-19-a", "IND-19-2", "IND-19-a-2"],
        )

    def test_classifies_only_explicit_right_categories(self) -> None:
        categories = classify_rights(
            "Freedom of speech", "Everyone has freedom of expression."
        )

        self.assertEqual(categories, ["freedom of expression"])

    def test_right_terms_do_not_match_inside_unrelated_words(self) -> None:
        self.assertEqual(classify_rights("Framework", "No relevant provision."), [])


if __name__ == "__main__":
    unittest.main()
