import json
import tempfile
import unittest
from pathlib import Path

from score_official import read_predictions, score, summarize


class OfficialTests(unittest.TestCase):
    def test_paired_scores_and_task_order(self):
        tasks = [dict(nl=str(i), bash="a", bash2="b", difficulty=i % 3) for i in range(4)]
        commands = ["a", "", "b", "c"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "predictions")
            path.write_text("".join(json.dumps(dict(nl=t["nl"], references=["a", "b"], command=c))+"\n"
                                    for t, c in zip(tasks, commands)))
            self.assertEqual(read_predictions(path, tasks), commands)
            with self.assertRaises(ValueError):
                read_predictions(path, tasks[::-1])
        calls = []

        def submit(index, command, eval_mode):
            calls.append(index)
            return int(command == "a")
        rows = score(tasks, commands, submit, {"eval_mode": "tfidf"}, ["a", "a", "a", "c"])
        self.assertEqual(calls, [0, 1, 2, 2, 3])
        result = summarize(rows)
        self.assertEqual((result["passed"], result["wins"], result["losses"]), (1, 0, 2))
        self.assertEqual(result["exact_mcnemar_p"], 0.5)

    def test_upstream_failure_is_not_counted_as_wrong(self):
        def failed(**kwargs):
            print("Benchmark Error: Docker unavailable")
            return 0
        with self.assertRaisesRegex(RuntimeError, "task 0"):
            score([{"difficulty": 0}], ["ls"], failed, {})
