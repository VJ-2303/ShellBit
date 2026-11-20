import unittest

from command_memory import CommandMemory
from intent_pairs import composed_examples, fixture_result


class MemoryTests(unittest.TestCase):
    def test_unseen_composition(self):
        rows = list(composed_examples("train"))
        memory = CommandMemory(rows)
        self.assertTrue(all(not (r["ordering"][0] and r["ordering"][1]) for r in rows))
        for row in composed_examples("dev"):
            self.assertIs(fixture_result(memory.translate(row["nl"]), row), True)

    def test_fallback_and_ambiguous_evidence(self):
        rows = list(composed_examples("train"))
        row = rows[0]
        memory = CommandMemory([row, {**row, "bash": rows[1]["bash"]}])
        self.assertIsNone(memory.translate(row["nl"]))
        self.assertIsNone(memory.translate("List all running processes."))
        self.assertIsNone(memory.translate(row["nl"].replace(row["filename"], "a;bad.txt")))
        self.assertIsNone(memory.translate("Also remove duplicates. " + row["nl"]))
