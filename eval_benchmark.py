#!/usr/bin/env python3
"""Evaluate reference matches, fixture outcomes, routing coverage and decoding latency."""

import argparse
import json
import math
from statistics import median
import shutil
from intent_pairs import fixture_result
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from command_memory import CommandMemory
from build_schema_rag import SchemaRAGRetriever
from infer import generate_verified_command
from train_rgrpo import load_sft_model
from verifiers import bash_token_f1, extract_primary_binaries


def main():
    parser = argparse.ArgumentParser(description="Evaluate ShellForge-135M on NL2SH-ALFA test split")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--test-file", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Optional subset size for quick eval")
    parser.add_argument("--k", type=int, default=1, help="Best-of-K verifier candidates")
    parser.add_argument("--predictions", help="Write per-prompt predictions to a new JSONL file")
    parser.add_argument("--collect-train", metavar="OUTPUT.jsonl",
                        help="Sample the prepared training fixtures and export verified alternatives")
    parser.add_argument("--command-memory", action="store_true", help="Evaluate fixture-memory + greedy model")
    args = parser.parse_args()
    if args.command_memory and (args.k != 1 or args.collect_train):
        parser.error("--command-memory requires k=1 and is unavailable during training-data collection")
    if args.collect_train and args.test_file:
        parser.error("--collect-train reads only fixture_train.jsonl")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ckpt_dir = args.ckpt or cfg["sft"]["save_dir"]
    test_file = args.test_file or str(Path(cfg["project"]["data_dir"]) / "test.jsonl")

    if args.collect_train:
        test_file = Path(cfg["project"]["data_dir"]) / "fixture_train.jsonl"
    predictions = []
    accepted = []
    torch.manual_seed(cfg["project"]["seed"])
    rows = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]

    if not rows or args.k < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("Evaluation needs a nonempty dataset and positive limit and k")
    paired = all("family" in row and "side" in row for row in rows)
    pairs = {}
    supported = correct = oracle = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_sft_model(ckpt_dir, device, use_bf16=(device.type == "cuda"))

    schema_db = cfg["project"]["schema_db"]
    retriever = SchemaRAGRetriever(schema_db) if cfg["schema_rag"]["enabled"] else None

    memory = CommandMemory.load(Path(cfg["project"]["data_dir"]) / "fixture_train.jsonl") if args.command_memory else None
    routes = []
    n = len(rows)
    syntax_pass = 0
    sc_clean = 0
    first_bin_match = 0
    exact_match = 0
    f1_sum = 0.0
    latencies = []

    for row in tqdm(rows, desc=f"Evaluating (Best-of-{args.k})"):
        candidates = []
        res, ms = generate_verified_command(
            row["nl"], model, tok, retriever, device, num_candidates=args.k, candidates=candidates,
            command_memory=memory, routes=routes,
            include_plan=cfg["data"]["include_compact_plan"],
            **{key: cfg["schema_rag"][key] for key in ("top_k_tools", "max_schema_chars")},
            **{key: cfg["inference"][key] for key in ("temperature", "top_p", "max_new_tokens")},
        )
        gold = row["bash"].strip()
        pred = "" if res.safety_level == "BLOCKED" else res.command.strip()

        if res.syntax_ok:
            syntax_pass += 1
        if res.syntax_ok and res.shellcheck_errors == 0 and res.shellcheck_warnings == 0:
            sc_clean += 1
        refs = {gold, row.get("bash2") or gold}
        if paired:
            outcomes = {cmd: fixture_result(cmd, row) for cmd in set(candidates)}
            if args.collect_train:
                accepted.extend({"nl": row["nl"], "bash": cmd} for cmd, passed in sorted(outcomes.items())
                                if passed and cmd != gold)
            result = outcomes.get(pred)
            correct += result is True
            supported += result is not None
            oracle += True in outcomes.values()
            pairs.setdefault((row["family"], row["filename"]), {})[row["side"]] = result
        if pred in refs:
            exact_match += 1

        p_bins = extract_primary_binaries(pred)
        g_bins = [b[0] for ref in refs if (b := extract_primary_binaries(ref))]
        if p_bins and g_bins and p_bins[0] in g_bins:
            first_bin_match += 1

        f1_sum += max(bash_token_f1(pred, ref) for ref in refs)
        latencies.append(ms)
        if args.predictions:
            prediction = {"source": routes[-1], "nl": row["nl"], "command": pred, "references": sorted(refs),
                          "difficulty": row.get("difficulty"), "latency_ms": ms}
            if paired:
                prediction.update(family=row["family"], side=row["side"], passed=result,
                                  any_pass=True in outcomes.values())
            predictions.append(prediction)

    if args.predictions:
        with open(args.predictions, "x", encoding="utf-8") as stream:
            for prediction in predictions:
                stream.write(json.dumps(prediction) + "\n")
    if args.collect_train:
        with open(args.collect_train, "x", encoding="utf-8") as stream:
            for row in accepted:
                stream.write(json.dumps(row) + "\n")
        print(f"Verified training alternatives: {len(accepted)}")
    if retriever:
        retriever.close()

    print("\n================ ShellForge-135M Benchmark Results ================")
    print(f"Checkpoint      : {ckpt_dir}")
    print(f"Samples         : {n} (Best-of-{args.k} verifier reranking)")
    print(f"SyntaxPass      : {100.0 * syntax_pass / max(1, n):.2f}%")
    if memory is not None:
        print(f"Command memory coverage: {routes.count('memory')}/{n} (others decoded by checkpoint)")
    if paired:
        complete = [v for v in pairs.values() if set(v) == {0, 1}]
        print(f"Fixture pass (selected): {correct}/{n}, any of {args.k}: {oracle}/{n}, supported: {supported}/{n}")
        print(f"Both requests correct: {sum(all(v.values()) for v in complete)}/{len(complete)} complete pairs")
    if shutil.which("shellcheck"):
        print(f"ShellCheckClean : {100.0 * sc_clean / max(1, n):.2f}%")
    print(f"FirstBinaryAcc  : {100.0 * first_bin_match / max(1, n):.2f}%")
    print(f"TokenF1         : {100.0 * f1_sum / max(1, n):.2f}%")
    print(f"ExactMatch      : {100.0 * exact_match / max(1, n):.2f}%")
    p95 = sorted(latencies)[math.ceil(0.95 * n) - 1]
    print(f"Latency (ms/cmd): mean={sum(latencies)/n:.1f} median={median(latencies):.1f} p95={p95:.1f}")
    print("===================================================================")


if __name__ == "__main__":
    main()
