#!/usr/bin/env python3
"""Generate Bash suggestions using greedy decoding or optional static reranking."""

import argparse
import os
import time
from typing import Tuple

import torch
import yaml

from command_memory import CommandMemory
from build_schema_rag import SchemaRAGRetriever
from model_arch import RecShellModel
from prepare_data import format_prompt
from train_rgrpo import load_sft_model
from verifiers import VerificationResult, compute_composite_reward


def generate_verified_command(
    nl_query: str,
    model,
    tokenizer,
    retriever: SchemaRAGRetriever,
    device: torch.device,
    num_candidates: int = 1,
    temperature: float = 0.4,
    top_p: float = 0.92,
    max_new_tokens: int = 140,
    include_plan: bool = True,
    top_k_tools: int = 3,
    max_schema_chars: int = 420,
    candidates=None,
    command_memory=None,
    routes=None,
) -> Tuple[VerificationResult, float]:
    if num_candidates < 1:
        raise ValueError("num_candidates must be positive")
    t0 = time.perf_counter()
    if command_memory is not None:
        if num_candidates != 1:
            raise ValueError("Command memory requires greedy decoding (k=1)")
        command = command_memory.translate(nl_query)
        if command is not None:
            result = compute_composite_reward(f"<cmd>{command}</cmd>", enable_sandbox_exec=False)
            if result.syntax_ok and result.safety_level == "SAFE":
                if candidates is not None:
                    candidates.append(result.command)
                if routes is not None:
                    routes.append("memory")
                return result, (time.perf_counter() - t0) * 1000
    if routes is not None:
        routes.append("neural")
    schema_card, _tools = retriever.retrieve(nl_query, top_k=top_k_tools, max_chars=max_schema_chars) if retriever else ("", [])
    prompt = format_prompt(nl_query, schema_card, include_plan)

    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = prompt_ids.size(1)
    batch_ids = prompt_ids.repeat(num_candidates, 1)

    model.eval()
    with torch.no_grad():
        if isinstance(model, RecShellModel):
            out_ids = model.generate(
                batch_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature if num_candidates > 1 else 0.0,
                top_p=top_p,
                eos_token_id=tokenizer.eos_token_id,
            )
        else:
            out_ids = model.generate(
                batch_ids,
                attention_mask=torch.ones_like(batch_ids),
                max_new_tokens=max_new_tokens,
                do_sample=(num_candidates > 1),
                temperature=temperature if num_candidates > 1 else None,
                top_p=top_p if num_candidates > 1 else None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

    best: VerificationResult = None
    for i in range(num_candidates):
        raw = tokenizer.decode(out_ids[i, prompt_len:], skip_special_tokens=True)
        res = compute_composite_reward(
            raw_output=raw,
            gold_cmd=None,
            schema_text=schema_card,
            enable_sandbox_exec=False,
        )
        if candidates is not None:
            candidates.append(res.command)
        if best is None or res.total_reward > best.total_reward:
            best = res

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return best, elapsed_ms


def main():
    parser = argparse.ArgumentParser(description="ShellForge-135M Verified NL-to-Bash CLI")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint dir (defaults to SFT checkpoint)")
    parser.add_argument("--query", type=str, default=None, help="Single NL query (else interactive REPL)")
    parser.add_argument("--k", type=int, default=None, help="Best-of-K verifier candidates")
    parser.add_argument("--command-memory", action="store_true", help="Use training-fixture memory before greedy generation")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ckpt_dir = args.ckpt or cfg["sft"]["save_dir"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda"

    print(f"[ShellForge] Loading {ckpt_dir} on {device}...")
    model, tok = load_sft_model(ckpt_dir, device, use_bf16)

    schema_db = cfg["project"]["schema_db"]
    retriever = SchemaRAGRetriever(schema_db) if cfg["schema_rag"]["enabled"] else None
    k = args.k or cfg["inference"]["num_candidates"]

    memory = CommandMemory.load(os.path.join(cfg["project"]["data_dir"], "fixture_train.jsonl")) if args.command_memory else None

    def run_one(q: str):
        routes = []
        res, ms = generate_verified_command(
            q,
            model,
            tok,
            retriever,
            device,
            include_plan=cfg["data"]["include_compact_plan"],
            **{key: cfg["schema_rag"][key] for key in ("top_k_tools", "max_schema_chars")},
            command_memory=memory, routes=routes, num_candidates=k,
            temperature=cfg["inference"]["temperature"],
            top_p=cfg["inference"]["top_p"],
            max_new_tokens=cfg["inference"]["max_new_tokens"],
        )
        print(f"\n[Source: {routes[0]} | Safety: {res.safety_level} | Syntax: {'OK' if res.syntax_ok else 'FAIL'} | {ms:.1f} ms]")
        if res.plan:
            print(f"Plan: {res.plan}")
        if res.safety_level == "BLOCKED":
            print("BLOCKED: Command matched a catastrophic safety rule and was withheld.")
        else:
            if res.safety_level == "CONFIRM_DESTRUCTIVE":
                print("WARNING: Destructive operation detected — inspect carefully before running!")
            print(f"Command:\n  {res.command}")

    if args.query:
        run_one(args.query)
    else:
        print("Interactive mode (type 'exit' to quit):")
        while True:
            try:
                q = input("\nnl2bash> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q.lower() in {"exit", "quit"}:
                break
            run_one(q)

    if retriever:
        retriever.close()


if __name__ == "__main__":
    main()
