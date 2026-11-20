#!/usr/bin/env python3
"""
smoke_test.py — Fast Offline CPU Verification Suite for ShellForge-135M.

Verifies every component without requiring internet access or a GPU:
  1. Local SQLite FTS5 Schema-RAG indexing & retrieval
  2. Verifiers (`bash -n`, `shellcheck`/heuristic, schema grounding, safety gate, composite reward)
  3. Data preparation formatting & compact `<plan>` synthesis
  4. `BitLinear158` ternary weight quantization + STE gradients
  5. `RecShellModel` recursive-depth forward + `SoftMTPWrapper` KL self-distillation loss + backward
  6. `rgrpo_loss_step` clipped policy gradient + Schulman KL backward
"""

import os
import tempfile

import torch

from build_schema_rag import SchemaRAGRetriever, build_schema_db
from model_arch import BitLinear158, RecShellConfig, RecShellModel, SoftMTPWrapper, count_parameters
from prepare_data import format_prompt, format_target, process_records, synthesize_compact_plan
from train_rgrpo import compute_token_logprobs, rgrpo_loss_step
from verifiers import check_safety, compute_composite_reward


def test_schema_rag():
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test_schema.sqlite")
        n = build_schema_db(db, harvest_local=False)
        assert n >= 45, f"Expected >=45 curated tools, got {n}"
        ret = SchemaRAGRetriever(db)
        card, tools = ret.retrieve("find all python files larger than 10MB and search for TODO", top_k=3)
        ret.close()
        assert "find" in tools, f"Expected 'find' in retrieved tools, got {tools}"
        assert len(card) > 20
        print(f"[PASS] Schema-RAG: indexed {n} tools, retrieved {tools}")


def test_verifiers_and_safety():
    good = "<plan>Use find with -name and -mtime; preserve safe quoting.</plan>\n<cmd>find /var/log -type f -name '*.log' -mtime +7</cmd>"
    res_good = compute_composite_reward(
        good,
        gold_cmd="find /var/log -type f -name '*.log' -mtime +7",
        schema_text="find: -name -type -mtime",
        enable_sandbox_exec=False,
    )
    assert res_good.syntax_ok, f"Expected syntax_ok=True: {res_good}"
    assert res_good.safety_level == "SAFE"
    assert res_good.total_reward >= 0.85, f"Expected high reward, got {res_good.total_reward}"

    bad_syntax = "<plan>broken</plan>\n<cmd>find /var/log -name '*.log</cmd>"
    res_bad = compute_composite_reward(bad_syntax, enable_sandbox_exec=False)
    assert not res_bad.syntax_ok

    blocked = "<plan>wipe</plan>\n<cmd>rm -rf /</cmd>"
    res_blk = compute_composite_reward(blocked, enable_sandbox_exec=False)
    assert res_blk.safety_level == "BLOCKED"
    assert res_blk.total_reward < 0

    assert check_safety("rm -rf ./build") == "CONFIRM_DESTRUCTIVE"
    print(f"[PASS] Verifiers: good_r={res_good.total_reward}, bad_syntax=caught, blocked=caught")


def test_data_prep():
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "schema.sqlite")
        build_schema_db(db, harvest_local=False)
        ret = SchemaRAGRetriever(db)
        raw = [
            {"nl": "show disk usage of current directory", "bash": "du -sh ."},
            {"nl": "broken command", "bash": "echo 'unterminated"},
            {"nl": "wipe root", "bash": "rm -rf /"},
        ]
        out = process_records(raw, retriever=ret, require_bash_n=True, require_shellcheck=False)
        ret.close()
        assert len(out) == 1, f"Expected 1 clean record, got {len(out)}"
        assert "<plan>" in out[0]["target"] and "<cmd>du -sh .</cmd>" in out[0]["target"]
        assert "[SCHEMA]" in out[0]["prompt"]
        print("[PASS] Data Prep: filtered syntax/blocked records, injected Schema-RAG & <plan>")


def test_recshell_bitnet_soft_mtp():
    # Tiny config for fast CPU smoke test
    cfg = RecShellConfig(
        vocab_size=256,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        n_physical_layers=2,
        n_recursions=3,
        ffn_dim=128,
        max_seq_len=64,
        use_bitnet_158=True,
    )
    base = RecShellModel(cfg)
    wrapper = SoftMTPWrapper(base, hidden_size=cfg.d_model, horizon=3, mtp_loss_weight=0.3)

    # Verify BitLinear158 ternary values
    bl = base.layers[0].attn.q_proj
    assert isinstance(bl, BitLinear158)
    wq = bl.quantize_weights_158(bl.weight)
    scale = bl.weight.abs().mean().clamp(min=1e-5)
    unique_vals = torch.unique((wq / scale).round())
    assert all(v.item() in (-1.0, 0.0, 1.0) for v in unique_vals), f"Non-ternary weights: {unique_vals}"

    # Verify Soft-MTP forward + backward
    ids = torch.randint(0, 256, (2, 24))
    labels = ids.clone()
    labels[:, :8] = -100  # prompt masking
    loss, stats = wrapper.forward_with_soft_mtp(ids, labels)
    loss.backward()
    assert torch.isfinite(loss), f"Non-finite loss: {loss}"
    assert stats["mtp_loss"] > 0.0
    total_p, _ = count_parameters(wrapper)
    print(f"[PASS] RecShell + BitNet158 + Soft-MTP: params={total_p:,}, ntp={stats['ntp_loss']:.3f}, mtp={stats['mtp_loss']:.3f}")


def test_rgrpo_math():
    cfg = RecShellConfig(
        vocab_size=128,
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        n_physical_layers=2,
        n_recursions=2,
        ffn_dim=64,
        max_seq_len=32,
        use_bitnet_158=False,
    )
    policy = RecShellModel(cfg)
    ref = RecShellModel(cfg)
    ref.load_state_dict(policy.state_dict())

    full_ids = torch.randint(1, 128, (4, 16))
    with torch.no_grad():
        old_lp, _ = compute_token_logprobs(policy, full_ids, prompt_len=6, pad_token_id=0)
    adv = torch.tensor([1.2, -0.8, 0.4, -0.8])
    loss, kl = rgrpo_loss_step(policy, ref, full_ids, prompt_len=6, old_logprobs=old_lp, advantages=adv, pad_token_id=0)
    loss.backward()
    assert torch.isfinite(loss), f"Non-finite R-GRPO loss: {loss}"
    assert abs(kl) < 1e-4, f"Initial KL vs identical ref should be ~0, got {kl}"
    print(f"[PASS] R-GRPO Math: loss={loss.item():.4f}, init_kl={kl:.6f}")


if __name__ == "__main__":
    test_schema_rag()
    test_verifiers_and_safety()
    test_data_prep()
    test_recshell_bitnet_soft_mtp()
    test_rgrpo_math()
    print("\nALL SHELLFORGE-135M SMOKE TESTS PASSED!")
