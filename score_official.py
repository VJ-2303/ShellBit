"""Score a complete ShellForge prediction file using upstream InterCode-ALFA."""

import argparse
import json
from collections import Counter
from contextlib import redirect_stdout
from io import StringIO
from importlib.metadata import version
from math import comb
from pathlib import Path


def read_predictions(path, tasks):
    with open(path, encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) != len(tasks):
        raise ValueError(f"Expected {len(tasks)} predictions, found {len(rows)}")
    commands = []
    for i, (row, task) in enumerate(zip(rows, tasks)):
        if row["nl"] != task["nl"] or set(row["references"]) != {task["bash"], task["bash2"]}:
            raise ValueError(f"Prediction row {i} differs from the official task order or references")
        command = row["command"]
        if not isinstance(command, str):
            raise ValueError(f"Invalid command at row {i}")
        commands.append(command)
    return commands


def score(tasks, commands, submit, judge, baseline=None):
    details = []
    for index, (task, command) in enumerate(zip(tasks, commands)):
        def evaluate(candidate):
            if not candidate.strip():
                return 0
            # Upstream prints infrastructure exceptions and returns a zero.
            # An unavailable judge must not become a measured model failure.
            log = StringIO()
            with redirect_stdout(log):
                result = submit(index=index, command=candidate, **judge)
            if any(line.startswith("Benchmark Error:") for line in log.getvalue().splitlines()):
                raise RuntimeError(f"InterCode-ALFA reported an evaluation error on task {index}")
            if result not in (0, 1):
                raise ValueError(f"Invalid judge result for task {index}: {result!r}")
            return int(result)

        passed = evaluate(command)
        row = {"index": index, "difficulty": task["difficulty"], "passed": passed}
        if baseline is not None:
            other = baseline[index]
            row["baseline_passed"] = passed if other == command else evaluate(other)
        details.append(row)
    return details


def summarize(rows):
    wins = losses = 0
    levels = {}
    for row in rows:
        level = levels.setdefault(str(row["difficulty"]), Counter())
        level["passed"] += row["passed"]
        level["tasks"] += 1
        if "baseline_passed" in row:
            wins += row["passed"] > row["baseline_passed"]
            losses += row["passed"] < row["baseline_passed"]
    summary = {"passed": sum(row["passed"] for row in rows), "tasks": len(rows), "by_difficulty": levels}
    if "baseline_passed" in rows[0]:
        discordant = wins + losses
        p = 2 * sum(comb(discordant, k) for k in range(min(wins, losses) + 1)) / 2**discordant
        summary.update(baseline_passed=sum(r["baseline_passed"] for r in rows),
                       wins=wins, losses=losses, delta_percentage_points=100*(wins-losses)/len(rows),
                       exact_mcnemar_p=min(1.0, p))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--baseline", type=Path, help="Matched predictions from a second checkpoint")
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--judge", required=True, choices=("tfidf", "ollama", "openai"))
    parser.add_argument("--judge-model", help="Explicit Ollama or OpenAI judge model identifier")
    parser.add_argument("--dataset-revision", help="Optional Hugging Face dataset commit")
    args = parser.parse_args()
    if (args.judge != "tfidf") != bool(args.judge_model):
        parser.error("--judge-model is required for ollama or openai, and unavailable for tfidf")
    if args.report.exists():
        parser.error("Report path already exists")
    from datasets import load_dataset
    from icalfa import submit_command

    tasks = load_dataset("westenfelder/NL2SH-ALFA", "test", split="train", revision=args.dataset_revision)
    if len(tasks) != 300:
        raise ValueError(f"Unexpected official prompt count: {len(tasks)}")
    commands = read_predictions(args.predictions, tasks)
    baseline = read_predictions(args.baseline, tasks) if args.baseline else None
    judge = {"eval_mode": args.judge}
    if args.judge_model:
        judge["eval_param"] = args.judge_model
    rows = score(tasks, commands, submit_command, judge, baseline)
    summary = summarize(rows)
    report = dict(judge=judge, icalfa_version=version("icalfa"), dataset_revision=args.dataset_revision,
                  dataset_fingerprint=tasks._fingerprint, predictions=str(args.predictions),
                  baseline=str(args.baseline) if args.baseline else None, summary=summary, results=rows)
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
