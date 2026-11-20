#!/usr/bin/env python3
"""Prepare NL2SH-ALFA splits, paired training fixtures and the development challenge."""

import argparse
import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

from intent_pairs import composed_examples, examples, fixture_result, rejected_command

import yaml
from tqdm import tqdm

from build_schema_rag import SchemaRAGRetriever, build_schema_db
from verifiers import (
    check_bash_syntax,
    check_safety,
    check_shellcheck,
    extract_primary_binaries,
)


SYSTEM_PROMPT = (
    "You are ShellForge, a precision Linux Bash command generator. "
    "Use the provided [SCHEMA] flags to output a brief <plan> and an exact, "
    "ShellCheck-clean <cmd>."
)


def format_prompt(nl_query: str, schema_card: str = "", include_plan: bool = True) -> str:
    """Build the unified prompt format shared across SFT, R-GRPO, and Inference."""
    instruction = SYSTEM_PROMPT if include_plan else "Return one Bash command enclosed in <cmd> and </cmd>."
    parts = [f"<|system|>\n{instruction}"]
    if schema_card:
        parts.append(f"[SCHEMA]\n{schema_card}\n[/SCHEMA]")
    parts.append(f"<|user|>\n{nl_query.strip()}\n<|assistant|>\n")
    return "\n".join(parts)


def synthesize_compact_plan(nl_query: str, bash_cmd: str) -> str:
    """
    Generate a concise (8-25 word) structural plan for a verified Bash command.
    Teaches a sub-150M model to decompose pipelines, filters, and safety checks
    before emitting tokens — without wasting inference latency on long CoT.
    """
    bins = extract_primary_binaries(bash_cmd)
    safety = check_safety(bash_cmd)
    has_pipe = "|" in bash_cmd and "||" not in bash_cmd
    has_chain = "&&" in bash_cmd or ";" in bash_cmd
    has_redir = bool(re.search(r"(?:>>?|<|2>&1)", bash_cmd))
    flags = re.findall(r"(?<!\w)(--[a-zA-Z0-9_-]+|-[a-zA-Z0-9]+)", bash_cmd)[:5]

    steps = []
    if bins:
        if has_pipe and len(bins) > 1:
            steps.append(f"Pipe {' -> '.join(bins[:4])}")
        elif has_chain and len(bins) > 1:
            steps.append(f"Chain {' then '.join(bins[:3])}")
        else:
            steps.append(f"Use {bins[0]}")
    if flags:
        steps.append(f"flags {' '.join(flags)}")
    if has_redir:
        steps.append("apply I/O redirection")
    if safety == "CONFIRM_DESTRUCTIVE":
        steps.append("destructive op: quote args carefully")
    else:
        steps.append("preserve safe quoting")

    return "; ".join(steps) + "."


def format_target(bash_cmd: str, plan: Optional[str] = None, include_plan: bool = True) -> str:
    """Format assistant target completion."""
    cmd_clean = bash_cmd.strip()
    if include_plan and plan:
        return f"<plan>{plan.strip()}</plan>\n<cmd>{cmd_clean}</cmd>"
    return f"<cmd>{cmd_clean}</cmd>"


def load_raw_records(hf_dataset: str, extra_jsonl: Optional[str] = None):
    """Load NL2SH-ALFA from Hugging Face + optional custom JSONL."""
    from datasets import load_dataset

    print(f"[Data] Loading Hugging Face dataset: {hf_dataset}")


    def normalize_split(split_obj) -> List[Dict]:
        rows = []
        for item in split_obj:
            nl = item.get("nl") or item.get("instruction") or item.get("prompt") or ""
            bash = item.get("bash") or item.get("cmd") or item.get("command") or ""
            plan = item.get("plan") or ""
            if nl.strip() and bash.strip():
                rows.append({**item, "nl": nl.strip(), "bash": bash.strip(), "plan": plan.strip()})
        return rows

    if hf_dataset == "westenfelder/NL2SH-ALFA":
        train_rows, test_rows = (
            normalize_split(load_dataset(hf_dataset, name, split="train"))
            for name in ("train", "test")
        )
        if (len(train_rows), len(test_rows)) != (40639, 300):
            raise ValueError("Unexpected NL2SH-ALFA revision or missing rows")
        if any(not r.get("bash2") or r.get("difficulty") is None for r in test_rows):
            raise ValueError("NL2SH-ALFA test metadata missing")
    else:
        ds = load_dataset(hf_dataset)
        train_rows, test_rows = (normalize_split(ds[name]) for name in ("train", "test"))

    if extra_jsonl:
        print(f"[Data] Merging extra JSONL: {extra_jsonl}")
        with open(extra_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                nl = obj.get("nl") or obj.get("instruction") or ""
                bash = obj.get("bash") or obj.get("cmd") or ""
                if nl.strip() and bash.strip():
                    train_rows.append(
                        {**obj, "nl": nl.strip(), "bash": bash.strip(), "plan": obj.get("plan", "").strip()}
                    )

    return train_rows, test_rows


def process_records(
    records: List[Dict],
    retriever: Optional[SchemaRAGRetriever],
    top_k: int = 3,
    max_schema_chars: int = 420,
    require_bash_n: bool = True,
    require_shellcheck: bool = False,
    include_plan: bool = True,
    deduplicate: bool = True,
    preserve_test: bool = False,
) -> List[Dict]:
    """Filter, ground with Schema-RAG, and format NL-to-Bash pairs."""
    seen_hashes = set()
    processed = []
    stats = {"dropped_blocked": 0, "dropped_syntax": 0, "dropped_shellcheck": 0, "dropped_dup": 0}

    for rec in tqdm(records, desc="Filtering & Schema-RAG grounding"):
        nl = rec["nl"]
        cmd = rec["bash"]

        if deduplicate:
            h = hashlib.md5(f"{nl.lower()}||{cmd}".encode("utf-8")).hexdigest()
            if h in seen_hashes:
                stats["dropped_dup"] += 1
                continue
            seen_hashes.add(h)

        safety = check_safety(cmd)
        if safety == "BLOCKED" and not preserve_test:
            stats["dropped_blocked"] += 1
            continue

        if require_bash_n:
            ok, _ = check_bash_syntax(cmd)
            if not ok:
                stats["dropped_syntax"] += 1
                continue

        if require_shellcheck:
            errs, warns, _ = check_shellcheck(cmd)
            if errs > 0 or warns > 0:
                stats["dropped_shellcheck"] += 1
                continue

        schema_card, tools = (
            retriever.retrieve(nl, top_k=top_k, max_chars=max_schema_chars)
            if retriever is not None
            else ("", [])
        )
        plan = rec.get("plan") or (synthesize_compact_plan(nl, cmd) if include_plan else "")

        processed.append(
            {
                **rec,
                "nl": nl,
                "bash": cmd,
                "schema": schema_card,
                "tools": tools,
                "safety": safety,
                "prompt": format_prompt(nl, schema_card, include_plan),
                "target": format_target(cmd, plan, include_plan=include_plan),
            }
        )

    print(f"[Data] Kept {len(processed)}/{len(records)} records | stats={stats}")
    return processed


def save_jsonl(path: str, rows: List[Dict]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare ShellForge-135M NL-to-Bash dataset")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--max-samples", type=int, default=None, help="Override max train samples")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    random.seed(cfg["project"]["seed"])
    data_dir = Path(cfg["project"]["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)

    schema_db = cfg["project"]["schema_db"]
    if cfg["schema_rag"]["enabled"]:
        if not os.path.exists(schema_db):
            build_schema_db(schema_db, harvest_local=True)
        retriever = SchemaRAGRetriever(schema_db)
    else:
        retriever = None

    train_raw, test_raw = load_raw_records(
        cfg["data"]["hf_dataset"], cfg["data"].get("extra_jsonl")
    )

    train_families = ("ordering", "frequency")
    fixture_rows = list(examples(train_families, ["values.txt", "sample.txt", "input.txt"]))
    challenge = list(examples(["endpoints", "counting"], ["items.txt", "records.txt"]))
    if cfg["data"].get("composed_fixtures"):
        fixture_rows.extend(composed_examples("train"))
        challenge.extend(composed_examples("dev"))
    held_out = {row["nl"].casefold() for row in test_raw + challenge}
    train_raw = [row for row in train_raw if row["nl"].casefold() not in held_out]
    max_n = args.max_samples or cfg["data"].get("max_train_samples")
    if max_n and len(train_raw) > max_n:
        random.shuffle(train_raw)
        train_raw = train_raw[:max_n]

    train_all = process_records(
        train_raw,
        retriever=retriever,
        top_k=cfg["schema_rag"]["top_k_tools"],
        max_schema_chars=cfg["schema_rag"]["max_schema_chars"],
        require_bash_n=cfg["data"]["require_bash_n_pass"],
        require_shellcheck=cfg["data"]["require_shellcheck_clean"],
        include_plan=cfg["data"]["include_compact_plan"],
        deduplicate=True,
    )

    # Keep alternate commands for a request and custom example families together.
    groups = {}
    for row in train_all:
        key = row.get("family") or row["nl"].casefold()
        groups.setdefault(key, []).append(row)
    if len(groups) < 2:
        raise ValueError("Need at least two distinct training groups")
    keys = list(groups)
    random.shuffle(keys)
    target_size = max(1, int(len(train_all) * cfg["data"]["val_split_ratio"]))
    val_rows, train_rows = [], []
    for key in keys[:-1]:
        destination = val_rows if len(val_rows) < target_size else train_rows
        destination.extend(groups[key])
    train_rows.extend(groups[keys[-1]])

    paired = cfg["data"].get("paired_training", False)
    reserved = held_out | {r["nl"].casefold() for r in val_rows + challenge}
    fixture_rows = [r for r in fixture_rows if r["nl"].casefold() not in reserved]
    partners = {(r["family"], r["filename"], r["side"]): r for r in fixture_rows}
    for row in fixture_rows:
        sibling = partners.get((row["family"], row["filename"], 1 - row["side"]))
        if sibling and not cfg["data"]["include_compact_plan"]:
            row["rejected_target"] = format_target(rejected_command(row, sibling), include_plan=False)
    fixture_train = process_records(
        fixture_rows, retriever,
        include_plan=cfg["data"]["include_compact_plan"],
        top_k=cfg["schema_rag"]["top_k_tools"], max_schema_chars=cfg["schema_rag"]["max_schema_chars"],
    )
    save_jsonl(str(data_dir / "fixture_train.jsonl"), fixture_train)
    if paired:
        train_rows.extend(fixture_train)
    if cfg["data"].get("verified_candidates"):
        by_prompt = {r["nl"]: r for r in fixture_train}
        with open(cfg["data"]["verified_candidates"], encoding="utf-8") as stream:
            accepted = []
            for line in stream:
                candidate = json.loads(line)
                row = by_prompt[candidate["nl"]]
                if fixture_result(candidate["bash"], row) is not True:
                    raise ValueError("Candidate failed its training fixture")
                accepted.append({**row, "bash": candidate["bash"]})
        train_rows.extend(process_records(
            accepted, retriever, include_plan=cfg["data"]["include_compact_plan"],
            top_k=cfg["schema_rag"]["top_k_tools"], max_schema_chars=cfg["schema_rag"]["max_schema_chars"],
        ))
    save_jsonl(str(data_dir / "challenge.jsonl"),
               challenge)

    # Process test split WITHOUT dropping valid benchmark items
    test_rows = process_records(
        test_raw,
        retriever=retriever,
        top_k=cfg["schema_rag"]["top_k_tools"],
        max_schema_chars=cfg["schema_rag"]["max_schema_chars"],
        require_bash_n=False,
        require_shellcheck=False,
        include_plan=cfg["data"]["include_compact_plan"],
        deduplicate=False,
        preserve_test=True,
    )

    save_jsonl(str(data_dir / "train.jsonl"), train_rows)
    save_jsonl(str(data_dir / "val.jsonl"), val_rows)
    save_jsonl(str(data_dir / "test.jsonl"), test_rows)

    if retriever is not None:
        retriever.close()

    print(
        f"[Data] Saved train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)} to {data_dir}/"
    )


if __name__ == "__main__":
    main()
