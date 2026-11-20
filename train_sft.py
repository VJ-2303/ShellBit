#!/usr/bin/env python3
"""Supervised command generation with optional auxiliary multi-token heads."""

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict

import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from functools import partial
from torch.nn.utils.rnn import pad_sequence

from model_arch import RecShellConfig, RecShellModel, SoftMTPWrapper, count_parameters


class BashSFTDataset(Dataset):
    """Prompt-masked causal LM dataset (loss computed ONLY on assistant tokens)."""

    def __init__(self, jsonl_path: str, tokenizer, max_seq_len: int = 512):
        if max_seq_len < 2:
            raise ValueError("max_seq_len must fit a prompt token and a target")
        self.samples = []
        skipped = 0
        with open(jsonl_path, encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                target = tokenizer.encode(row["target"], add_special_tokens=False) + [tokenizer.eos_token_id]
                if len(target) >= max_seq_len:
                    skipped += 1
                    continue
                prompt = tokenizer.encode(row["prompt"], add_special_tokens=True)
                prompt = prompt[-(max_seq_len - len(target)):]
                if not prompt:
                    raise ValueError("Empty encoded prompt")
                rejected = row.get("rejected_target")
                decision = (-1, -1)
                if rejected:
                    alternative = tokenizer.encode(rejected, add_special_tokens=False) + [tokenizer.eos_token_id]
                    for pos, (chosen, other) in enumerate(zip(target, alternative)):
                        if chosen != other:
                            decision = (len(prompt) + pos - 1, other)
                            break
                self.samples.append((torch.tensor(prompt + target), len(prompt), decision))
        if not self.samples:
            raise ValueError(f"No usable records in {jsonl_path}")
        print(f"[SFT] {jsonl_path}: {len(self.samples)} encoded, {skipped} overlength targets excluded")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_commands(samples, pad_token_id):
    sequences, prompt_lengths, decisions = zip(*samples)
    ids = pad_sequence(sequences, batch_first=True, padding_value=pad_token_id)
    positions = torch.arange(ids.size(1)).unsqueeze(0)
    mask = positions < torch.tensor([len(seq) for seq in sequences]).unsqueeze(1)
    supervised = mask & (positions >= torch.tensor(prompt_lengths).unsqueeze(1))
    return {"input_ids": ids, "attention_mask": mask.long(),
            "labels": ids.masked_fill(~supervised, -100), "decisions": torch.tensor(decisions)}


def build_model_and_tokenizer(cfg: Dict, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mode = cfg["model"]["mode"]
    tok = AutoTokenizer.from_pretrained(cfg["model"]["pretrained_id"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    if mode == "smollm2_135m":
        print(f"[SFT] Loading pretrained warm-start: {cfg['model']['pretrained_id']}")
        base = AutoModelForCausalLM.from_pretrained(
            cfg["model"]["pretrained_id"],
            torch_dtype=torch.float32,  # Keep optimizer updates in FP32 under autocast.
        )
        if cfg["sft"]["gradient_checkpointing"]:
            base.gradient_checkpointing_enable()
        hidden_size = base.config.hidden_size
    elif mode == "recshell_120m":
        rcfg_dict = cfg["model"]["recshell"]
        rcfg = RecShellConfig(
            vocab_size=len(tok),
            d_model=rcfg_dict["d_model"],
            n_heads=rcfg_dict["n_heads"],
            n_kv_heads=rcfg_dict["n_kv_heads"],
            n_physical_layers=rcfg_dict["n_physical_layers"],
            n_recursions=rcfg_dict["n_recursions"],
            ffn_dim=rcfg_dict["ffn_dim"],
            max_seq_len=cfg["model"]["max_seq_len"],
            use_bitnet_158=rcfg_dict["use_bitnet_158"],
            tie_word_embeddings=rcfg_dict["tie_word_embeddings"],
        )
        print(
            f"[SFT] Initializing RecShell-120M (layers={rcfg.n_physical_layers}x{rcfg.n_recursions}, "
            f"bitnet158={rcfg.use_bitnet_158})"
        )
        base = RecShellModel(rcfg)
        hidden_size = rcfg.d_model
    else:
        raise ValueError(f"Unknown model.mode: {mode}")

    mtp_cfg = cfg["model"]["soft_mtp"]
    wrapper = SoftMTPWrapper(
        base_model=base,
        hidden_size=hidden_size,
        horizon=mtp_cfg["horizon"] if mtp_cfg["enabled"] else 1,
        mtp_loss_weight=mtp_cfg["mtp_loss_weight"] if mtp_cfg["enabled"] else 0.0,
        temperature=mtp_cfg["temperature"],
    ).to(device)

    total_p, train_p = count_parameters(wrapper)
    if total_p >= 150_000_000:
        raise ValueError(f"Parameter limit exceeded: {total_p:,}")
    print(f"[SFT] Total params: {total_p / 1e6:.2f}M | Trainable: {train_p / 1e6:.2f}M")
    return wrapper, tok


@torch.no_grad()
def evaluate(wrapper: SoftMTPWrapper, loader: DataLoader, device: torch.device, use_bf16: bool) -> float:
    wrapper.eval()
    total_nll = 0.0
    total_tokens = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attn_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            _, stats = wrapper.forward_with_soft_mtp(input_ids, labels, attn_mask)
        tokens = int(batch["labels"][:, 1:].ne(-100).sum())
        total_nll += stats["ntp_loss"] * tokens
        total_tokens += tokens
    wrapper.train()
    return total_nll / total_tokens


def save_checkpoint(wrapper: SoftMTPWrapper, tokenizer, save_dir: str, cfg: Dict):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    if isinstance(wrapper.base_model, RecShellModel):
        torch.save(wrapper.base_model.state_dict(), os.path.join(save_dir, "recshell_model.pt"))
        with open(os.path.join(save_dir, "recshell_config.json"), "w") as f:
            json.dump(wrapper.base_model.cfg.__dict__, f, indent=2)
    else:
        wrapper.base_model.save_pretrained(save_dir)
    # Save Soft-MTP heads too for speculative inference or resumed training
    torch.save(wrapper.mtp_projs.state_dict(), os.path.join(save_dir, "soft_mtp_heads.pt"))
    tokenizer.save_pretrained(save_dir)
    with open(os.path.join(save_dir, "train_config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)
    print(f"[SFT] Saved checkpoint -> {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune the configured NL-to-Bash model")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["project"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = bool(cfg["sft"]["bf16"] and device.type == "cuda" and torch.cuda.is_bf16_supported())

    wrapper, tok = build_model_and_tokenizer(cfg, device)

    data_dir = Path(cfg["project"]["data_dir"])
    train_ds = BashSFTDataset(str(data_dir / "train.jsonl"), tok, cfg["model"]["max_seq_len"])
    val_ds = BashSFTDataset(str(data_dir / "val.jsonl"), tok, cfg["model"]["max_seq_len"])

    contrast_weight = cfg["sft"].get("contrast_weight", 0.0)
    if not math.isfinite(contrast_weight) or contrast_weight < 0:
        raise ValueError("contrast_weight must be nonnegative")
    if contrast_weight and not any(decision[0] >= 0 for _, _, decision in train_ds):
        raise ValueError("Positive contrast_weight requires training rows with distinct rejected targets")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["sft"]["batch_size"],
        shuffle=True,
        collate_fn=partial(collate_commands, pad_token_id=tok.pad_token_id),
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["sft"]["batch_size"],
        shuffle=False,
        collate_fn=partial(collate_commands, pad_token_id=tok.pad_token_id),
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # Prefer 8-bit AdamW if bitsandbytes is installed on CUDA, else fused AdamW
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            wrapper.parameters(),
            lr=cfg["sft"]["learning_rate"],
            weight_decay=cfg["sft"]["weight_decay"],
        ) if device.type == "cuda" else torch.optim.AdamW(
            wrapper.parameters(), lr=cfg["sft"]["learning_rate"], weight_decay=cfg["sft"]["weight_decay"]
        )
    except ImportError:
        optimizer = torch.optim.AdamW(
            wrapper.parameters(),
            lr=cfg["sft"]["learning_rate"],
            weight_decay=cfg["sft"]["weight_decay"],
            fused=(device.type == "cuda"),
        )

    grad_accum = cfg["sft"]["grad_accum_steps"]
    total_steps = math.ceil(len(train_loader) / grad_accum) * cfg["sft"]["epochs"]
    warmup_steps = max(1, int(total_steps * cfg["sft"]["warmup_ratio"]))
    min_lr_ratio = cfg["sft"]["min_lr_ratio"]

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(
        f"[SFT] Starting training on {device} | epochs={cfg['sft']['epochs']} | "
        f"opt_steps={total_steps} | bf16={use_bf16}"
    )
    wrapper.train()
    global_step = 0
    best_val = float("inf")
    t0 = time.time()
    supervised_tokens = padded_slots = input_tokens = 0

    for epoch in range(cfg["sft"]["epochs"]):
        optimizer.zero_grad(set_to_none=True)
        for micro_step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attn_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            input_tokens += int(batch["attention_mask"].sum())
            padded_slots += batch["input_ids"].numel()
            supervised_tokens += int(batch["labels"].ne(-100).sum())
            decisions = batch["decisions"] if contrast_weight else None
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                loss, stats = wrapper.forward_with_soft_mtp(input_ids, labels, attn_mask, decisions, contrast_weight)
                group_start = micro_step - micro_step % grad_accum
                loss = loss / min(grad_accum, len(train_loader) - group_start)

            loss.backward()

            if (micro_step + 1) % grad_accum == 0 or micro_step + 1 == len(train_loader):
                torch.nn.utils.clip_grad_norm_(wrapper.parameters(), cfg["sft"]["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % cfg["sft"]["log_every"] == 0:
                    vram_gb = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
                    elapsed = time.time() - t0
                    print(
                        f"epoch={epoch+1} step={global_step}/{total_steps} "
                        f"ntp={stats['ntp_loss']:.4f} soft_mtp={stats['mtp_loss']:.4f} "
                        f"target_tokens={supervised_tokens} batch_util={input_tokens/padded_slots:.1%} "
                        f"lr={scheduler.get_last_lr()[0]:.2e} vram={vram_gb:.2f}GB t={elapsed:.0f}s"
                    )

                if global_step % cfg["sft"]["eval_every"] == 0:
                    val_loss = evaluate(wrapper, val_loader, device, use_bf16)
                    print(f"[SFT] Eval @ step {global_step}: val_ntp_loss={val_loss:.4f}")
                    if val_loss < best_val:
                        best_val = val_loss
                        save_checkpoint(wrapper, tok, cfg["sft"]["save_dir"], cfg)

    val_loss = evaluate(wrapper, val_loader, device, use_bf16)
    print(f"[SFT] Final val_ntp_loss={val_loss:.4f} (best={best_val:.4f})")
    if val_loss <= best_val:
        save_checkpoint(wrapper, tok, cfg["sft"]["save_dir"], cfg)


if __name__ == "__main__":
    main()
