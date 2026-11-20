import unittest

from intent_pairs import FAMILIES, examples, fixture_result


class IntentPairTests(unittest.TestCase):
    def test_references_and_swapped_intents(self):
        for family in FAMILIES:
            left, right = examples([family], ["sample.txt"])
            for row, other in ((left, right), (right, left)):
                with self.subTest(family=family, side=row["side"]):
                    self.assertIs(fixture_result(row["bash"], row), True)
                    self.assertIs(fixture_result(other["bash"], row), False)

    def test_unsupported_and_alternative(self):
        row = next(examples(["endpoints"], ["items.txt"]))
        for cmd in ["head -1 wrong.txt", "head -1 items.txt > other", "cat /etc/hosts", ""]:
            self.assertIsNone(fixture_result(cmd, row))
        self.assertIs(fixture_result("cat items.txt | head -1", row), True)

    def test_compositions_separate_reference_from_partner(self):
        from intent_pairs import composed_examples
        families = {}
        for split in ("train", "dev"):
            rows = list(composed_examples(split))
            families[split] = {r["family"] for r in rows}
            for i, row in enumerate(rows):
                with self.subTest(split=split, nl=row["nl"]):
                    self.assertIs(fixture_result(row["bash"], row), True)
                    self.assertIs(fixture_result(rows[i ^ 1]["bash"], row), False)
        self.assertTrue(families["train"].isdisjoint(families["dev"]))
