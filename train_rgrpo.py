#!/usr/bin/env python3
"""Optional GRPO on training fixtures with binary functional rewards."""

import argparse
import copy
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import yaml

from model_arch import RecShellConfig, RecShellModel
from intent_pairs import fixture_result
from verifiers import compute_composite_reward


def load_sft_model(ckpt_dir: str, device: torch.device, use_bf16: bool):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ckpt_dir)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    recshell_cfg_path = os.path.join(ckpt_dir, "recshell_config.json")
    if os.path.exists(recshell_cfg_path):
        with open(recshell_cfg_path, "r") as f:
            rcfg = RecShellConfig(**json.load(f))
        model = RecShellModel(rcfg)
        model.load_state_dict(torch.load(os.path.join(ckpt_dir, "recshell_model.pt"), map_location="cpu"))
        model = model.to(device=device, dtype=(torch.bfloat16 if use_bf16 else torch.float32))
    else:
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_dir,
            torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        ).to(device)
    return model, tok


def compute_token_logprobs(
    model,
    full_ids: torch.Tensor,
    prompt_len: int,
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-token log-probabilities for generated completion tokens and mask."""
    if isinstance(model, RecShellModel):
        logits = model(full_ids)["logits"]
    else:
        attn_mask = (full_ids != pad_token_id).long()
        logits = model(input_ids=full_ids, attention_mask=attn_mask, use_cache=False).logits

    # Align logits at t with target token at t+1
    shift_logits = logits[:, prompt_len - 1 : -1, :]
    target_ids = full_ids[:, prompt_len:]

    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    token_lp = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    comp_mask = (target_ids != pad_token_id).float()
    return token_lp, comp_mask


def rgrpo_loss_step(
    policy_model,
    ref_model,
    full_ids: torch.Tensor,
    prompt_len: int,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    pad_token_id: int,
    kl_beta: float = 0.04,
    clip_eps: float = 0.2,
    use_bf16: bool = False,
) -> Tuple[torch.Tensor, float]:
    """Compute clipped GRPO policy gradient + Schulman unbiased KL estimator."""
    device_type = full_ids.device.type
    with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_bf16):
        new_lp, mask = compute_token_logprobs(policy_model, full_ids, prompt_len, pad_token_id)

    with torch.no_grad():
        ref_lp, _ = compute_token_logprobs(ref_model, full_ids, prompt_len, pad_token_id)

    ratio = torch.exp(new_lp - old_logprobs.detach())
    adv = advantages.unsqueeze(1)  # [G, 1]
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    pg_loss = -torch.min(surr1, surr2)

    # Unbiased per-token KL(policy || ref) = exp(ref - new) - (ref - new) - 1
    log_r = ref_lp - new_lp
    kl = torch.exp(log_r.clamp(-10.0, 10.0)) - log_r - 1.0

    per_tok = (pg_loss + kl_beta * kl) * mask
    loss = per_tok.sum() / mask.sum().clamp(min=1.0)
    mean_kl = (kl * mask).sum().item() / mask.sum().clamp(min=1.0).item()
    return loss, mean_kl


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Execution-Verified R-GRPO Trainer")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    random.seed(cfg["project"]["seed"])
    torch.manual_seed(cfg["project"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = bool(cfg["rgrpo"]["bf16"] and device.type == "cuda")

    sft_dir = cfg["sft"]["save_dir"]
    print(f"[R-GRPO] Loading SFT checkpoint from {sft_dir}")
    policy, tok = load_sft_model(sft_dir, device, use_bf16)
    ref_model = copy.deepcopy(policy).eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    if hasattr(policy, "gradient_checkpointing_enable"):
        policy.gradient_checkpointing_enable()

    # Load training prompts for RLVR
    train_path = Path(cfg["project"]["data_dir"]) / "fixture_train.jsonl"
    prompts_pool: List[Dict] = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts_pool.append(json.loads(line))
    print(f"[R-GRPO] Loaded {len(prompts_pool)} verifiable RL prompts")

    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg["rgrpo"]["learning_rate"], weight_decay=0.01)

    rcfg = cfg["rgrpo"]
    G = rcfg["group_size"]
    if not prompts_pool or G < 2:
        raise ValueError("Fixture RL requires prepared training fixtures and group_size >= 2")
    save_dir = rcfg["save_dir"]
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for step in range(1, rcfg["steps"] + 1):
        batch_items = random.sample(prompts_pool, k=min(rcfg["prompts_per_step"], len(prompts_pool)))
        optimizer.zero_grad(set_to_none=True)

        step_rewards = []
        step_syntax = []
        step_sc_clean = []
        step_kls = []

        for item in batch_items:
            prompt_ids = tok.encode(item["prompt"], return_tensors="pt").to(device)
            prompt_len = prompt_ids.size(1)
            group_prompts = prompt_ids.repeat(G, 1)

            # 1. Sample G rollouts
            policy.eval()
            with torch.no_grad():
                if isinstance(policy, RecShellModel):
                    full_ids = policy.generate(
                        group_prompts,
                        max_new_tokens=rcfg["max_new_tokens"],
                        temperature=rcfg["temperature"],
                        top_p=rcfg["top_p"],
                        eos_token_id=tok.eos_token_id,
                    )
                else:
                    full_ids = policy.generate(
                        group_prompts,
                        attention_mask=torch.ones_like(group_prompts),
                        max_new_tokens=rcfg["max_new_tokens"],
                        do_sample=True,
                        temperature=rcfg["temperature"],
                        top_p=rcfg["top_p"],
                        pad_token_id=tok.pad_token_id,
                        eos_token_id=tok.eos_token_id,
                    )
                old_lp, _ = compute_token_logprobs(policy, full_ids, prompt_len, tok.pad_token_id)

            # 2. Score each rollout with deterministic verifiers
            rewards = []
            for g in range(G):
                comp_text = tok.decode(full_ids[g, prompt_len:], skip_special_tokens=True)
                vres = compute_composite_reward(
                    raw_output=comp_text,
                    gold_cmd=item["bash"],
                    schema_text=item.get("schema", ""),

                    enable_sandbox_exec=False,
                )
                rewards.append(float(fixture_result(vres.command, item) is True))
                step_syntax.append(1.0 if vres.syntax_ok else 0.0)
                step_sc_clean.append(
                    1.0 if (vres.syntax_ok and vres.shellcheck_errors == 0 and vres.shellcheck_warnings == 0) else 0.0
                )

            r_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
            step_rewards.extend(rewards)

            # Skip zero-variance groups (no learning signal)
            if r_tensor.std().item() < 1e-5:
                continue

            advantages = (r_tensor - r_tensor.mean()) / (r_tensor.std() + 1e-6)

            # 3. Clipped GRPO policy update
            policy.train()
            loss, mean_kl = rgrpo_loss_step(
                policy_model=policy,
                ref_model=ref_model,
                full_ids=full_ids,
                prompt_len=prompt_len,
                old_logprobs=old_lp,
                advantages=advantages,
                pad_token_id=tok.pad_token_id,
                kl_beta=rcfg["kl_beta"],
                clip_eps=rcfg["clip_eps"],
                use_bf16=use_bf16,
            )
            (loss / len(batch_items)).backward()
            step_kls.append(mean_kl)

        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        if step % 10 == 0 or step == 1:
            avg_r = sum(step_rewards) / max(1, len(step_rewards))
            avg_syn = 100.0 * sum(step_syntax) / max(1, len(step_syntax))
            avg_sc = 100.0 * sum(step_sc_clean) / max(1, len(step_sc_clean))
            avg_kl = sum(step_kls) / max(1, len(step_kls))
            vram = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
            print(
                f"[R-GRPO] step={step}/{rcfg['steps']} reward={avg_r:.3f} "
                f"mixed_groups={len(step_kls)}/{len(batch_items)} syntax={avg_syn:.1f}% sc_clean={avg_sc:.1f}% kl={avg_kl:.4f} vram={vram:.2f}GB"
            )

        if step % 150 == 0 or step == rcfg["steps"]:
            if isinstance(policy, RecShellModel):
                torch.save(policy.state_dict(), os.path.join(save_dir, "recshell_model.pt"))
                with open(os.path.join(save_dir, "recshell_config.json"), "w") as f:
                    json.dump(policy.cfg.__dict__, f, indent=2)
            else:
                policy.save_pretrained(save_dir)
            tok.save_pretrained(save_dir)
            print(f"[R-GRPO] Saved checkpoint -> {save_dir}")


if __name__ == "__main__":
    main()
