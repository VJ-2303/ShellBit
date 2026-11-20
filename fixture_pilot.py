"""Offline controlled experiment on compositional fixtures using a tiny byte-level model."""

import argparse
import json
import random
import tempfile
from pathlib import Path

import torch

from command_memory import CommandMemory
from intent_pairs import composed_examples, fixture_result, rejected_command
from model_arch import RecShellConfig, RecShellModel, SoftMTPWrapper, count_parameters
from prepare_data import format_prompt, format_target
from train_sft import BashSFTDataset, collate_commands
from verifiers import extract_plan_and_cmd


class ByteTokenizer:
    eos_token_id = 256

    def encode(self, text, add_special_tokens=True):
        return list(text.encode("utf-8"))


def pilot_data(path, split):
    rows = list(composed_examples(split))
    partners = {(r["family"], r["filename"], r["side"]): r for r in rows}
    for row in rows:
        sibling = partners[row["family"], row["filename"], 1 - row["side"]]
        row.update(prompt=format_prompt(row["nl"], include_plan=False),
                   target=format_target(row["bash"]),
                   rejected_target=format_target(rejected_command(row, sibling)))
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows, BashSFTDataset(path, ByteTokenizer(), 512)


@torch.inference_mode()
def evaluate_pilot(model, rows, dataset):
    model.eval()
    batch = collate_commands(dataset.samples, 256)
    pos, alternative = batch["decisions"].T
    idx = torch.arange(len(rows))
    logits = model(batch["input_ids"])["logits"]
    choices = logits[idx, pos, batch["labels"][idx, pos + 1]] > logits[idx, pos, alternative]
    passed = 0
    supported = 0
    for row in rows:
        ids = torch.tensor([ByteTokenizer().encode(row["prompt"])])
        completion = model.generate(ids, max_new_tokens=90, temperature=0, eos_token_id=256)
        tokens = completion[0, ids.size(1):].tolist()
        raw = bytes(t for t in tokens if t < 256).decode("utf-8", errors="replace")
        _, cmd, _ = extract_plan_and_cmd(raw)
        outcome = fixture_result(cmd, row)
        passed += outcome is True
        supported += outcome is not None
    return {"preferred_verified_token": int(choices.sum()), "greedy_fixture_pass": passed,
            "supported": supported, "prompts": len(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=160)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as directory:
        train_rows, train = pilot_data(Path(directory, "train"), "train")
        rows, dev = pilot_data(Path(directory, "dev"), "dev")
        memory = CommandMemory(train_rows)
        commands = [memory.translate(r["nl"]) for r in rows]
        passed = sum(fixture_result(c, r) is True for c, r in zip(commands, rows) if c)
        print(json.dumps({"model": "command_memory", "prompts": len(rows), "functional_pass": passed,
                          "answered": sum(c is not None for c in commands)}))
        cfg = RecShellConfig(vocab_size=257, d_model=48, n_heads=4, n_kv_heads=2,
                             n_physical_layers=1, n_recursions=1, ffn_dim=96,
                             use_bitnet_158=False, tie_word_embeddings=False)
        for weight in (0.0, 0.2):
            torch.manual_seed(args.seed)
            rng = random.Random(args.seed)
            model = RecShellModel(cfg)
            wrapper = SoftMTPWrapper(model, cfg.d_model, horizon=1)
            optimizer = torch.optim.AdamW(wrapper.parameters(), lr=0.002)
            slots = 0
            for step in range(args.steps):
                batch = collate_commands(rng.sample(train.samples, 8), 256)
                loss, _ = wrapper.forward_with_soft_mtp(
                    batch["input_ids"], batch["labels"], batch["attention_mask"],
                    batch["decisions"], weight)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                slots += batch["input_ids"].numel()
            result = evaluate_pilot(model, rows, dev)
            print(json.dumps(dict(result, seed=args.seed, steps=args.steps, contrast_weight=weight,
                                  parameters=count_parameters(wrapper)[0], padded_slots=slots,
                                  fixed_512_slots=args.steps * 8 * 512)), flush=True)


if __name__ == "__main__":
    main()
