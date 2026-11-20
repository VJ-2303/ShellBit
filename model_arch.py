#!/usr/bin/env python3
"""Recursive transformer and auxiliary objectives for sub-150M command models."""

import inspect
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. Native 1.58-bit BitLinear Layer (BitNet b1.58 W1.58A8 with STE)
# ==============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class BitLinear158(nn.Linear):
    """
    BitNet b1.58 Linear Layer:
      - Weights quantized to {-1, 0, +1} using absmean scaling (STE in backward).
      - Activations quantized per-token to INT8 [-127, 127] using absmax scaling.
      - Includes SubLN pre-normalization for stable 1.58-bit gradient flow.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__(in_features, out_features, bias=bias)
        self.sub_ln = RMSNorm(in_features)

    @staticmethod
    def quantize_weights_158(w: torch.Tensor) -> torch.Tensor:
        scale = w.abs().mean().clamp(min=1e-5)
        w_q = (w / scale).round().clamp(-1.0, 1.0) * scale
        return w + (w_q - w).detach()  # Straight-Through Estimator (STE)

    @staticmethod
    def quantize_activations_int8(x: torch.Tensor) -> torch.Tensor:
        scale = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        x_q = (x * scale).round().clamp(-127.0, 127.0) / scale
        return x + (x_q - x).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.sub_ln(x)
        x_q = self.quantize_activations_int8(x_norm)
        w_q = self.quantize_weights_158(self.weight)
        return F.linear(x_q, w_q, self.bias)


def make_linear(in_dim: int, out_dim: int, use_bitnet: bool) -> nn.Module:
    if use_bitnet:
        return BitLinear158(in_dim, out_dim, bias=False)
    return nn.Linear(in_dim, out_dim, bias=False)


# ==============================================================================
# 2. RecShell-120M: Weight-Tied Recursive-Depth Transformer + Optional BitNet
# ==============================================================================

@dataclass
class RecShellConfig:
    vocab_size: int = 49152
    d_model: int = 768
    n_heads: int = 12
    n_kv_heads: int = 4
    n_physical_layers: int = 8
    n_recursions: int = 3          # 8 physical x 3 loops = 24 effective layers!
    ffn_dim: int = 2048
    max_seq_len: int = 512
    use_bitnet_158: bool = True
    tie_word_embeddings: bool = True
    rope_theta: float = 10000.0


def precompute_rope(dim: int, max_seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, D]
    T = x.size(2)
    cos = cos[:T].unsqueeze(0).unsqueeze(0).to(x.dtype)
    sin = sin[:T].unsqueeze(0).unsqueeze(0).to(x.dtype)
    x1, x2 = x[..., ::2], x[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2)


class GQAAttention(nn.Module):
    def __init__(self, cfg: RecShellConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads

        self.q_proj = make_linear(cfg.d_model, self.n_heads * self.head_dim, cfg.use_bitnet_158)
        self.k_proj = make_linear(cfg.d_model, self.n_kv_heads * self.head_dim, cfg.use_bitnet_158)
        self.v_proj = make_linear(cfg.d_model, self.n_kv_heads * self.head_dim, cfg.use_bitnet_158)
        self.o_proj = make_linear(self.n_heads * self.head_dim, cfg.d_model, cfg.use_bitnet_158)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


class Relu2GatedFFN(nn.Module):
    """BitNet-friendly ReLU^2 gated FFN (higher sparsity & cleaner 1.58-bit dynamics)."""

    def __init__(self, cfg: RecShellConfig):
        super().__init__()
        self.gate_proj = make_linear(cfg.d_model, cfg.ffn_dim, cfg.use_bitnet_158)
        self.up_proj = make_linear(cfg.d_model, cfg.ffn_dim, cfg.use_bitnet_158)
        self.down_proj = make_linear(cfg.ffn_dim, cfg.d_model, cfg.use_bitnet_158)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.relu(self.gate_proj(x)).pow(2)
        return self.down_proj(gate * self.up_proj(x))


class RecShellBlock(nn.Module):
    def __init__(self, cfg: RecShellConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = GQAAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = Relu2GatedFFN(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ffn(self.norm2(x))
        return x


class RecShellModel(nn.Module):
    """
    Weight-Tied Recursive-Depth Transformer:
      - Runs `n_physical_layers` blocks `n_recursions` times with learned per-loop
        step embeddings + residual anchor injection so latent representations
        refine iteratively (effective depth = n_physical_layers * n_recursions).
    """

    def __init__(self, cfg: RecShellConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.loop_embed = nn.Embedding(cfg.n_recursions, cfg.d_model)
        self.loop_gate = nn.Parameter(torch.full((cfg.n_recursions,), 0.25))

        self.layers = nn.ModuleList([RecShellBlock(cfg) for _ in range(cfg.n_physical_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        cos, sin = precompute_rope(cfg.d_model // cfg.n_heads, cfg.max_seq_len * 2, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = True,
    ) -> Dict[str, torch.Tensor]:
        h0 = self.embed_tokens(input_ids)
        h = h0
        for r in range(self.cfg.n_recursions):
            # Inject recursion step embedding + gated anchor skip from input embedding
            step_vec = self.loop_embed.weight[r].view(1, 1, -1)
            h = h + step_vec + self.loop_gate[r] * h0
            for layer in self.layers:
                h = layer(h, self.rope_cos, self.rope_sin)

        hidden = self.norm(h)
        logits = self.lm_head(hidden)
        return {"logits": logits, "last_hidden_state": hidden}

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.92,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        out = input_ids
        for _ in range(max_new_tokens):
            ctx = out[:, -self.cfg.max_seq_len :]
            logits = self.forward(ctx)["logits"][:, -1, :]
            if temperature <= 0.0:
                next_tok = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cum = torch.cumsum(sorted_probs, dim=-1)
                mask = cum - sorted_probs > top_p
                sorted_probs[mask] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                sampled = torch.multinomial(sorted_probs, num_samples=1)
                next_tok = sorted_idx.gather(-1, sampled)
            out = torch.cat([out, next_tok], dim=-1)
            if eos_token_id is not None and (next_tok == eos_token_id).all():
                break
        return out


# ==============================================================================
# 3. Soft-MTP Wrapper ("Multi-Token Prediction Done Right", arXiv:2610.38331)
# ==============================================================================

class SoftMTPWrapper(nn.Module):
    """
    Wraps any causal LM (SmolLM2-135M or RecShell-120M) with Soft Multi-Token
    Prediction (Soft-MTP):
      - Adds lightweight linear projection adapters for future offsets k = 2..H
        sharing the model's existing `lm_head` (adds <0.5% params!).
      - Computes the Soft-MTP KL self-distillation objective: instead of only
        hard one-hot targets for future tokens (which are noisy on small models),
        aligns future-head predictions with the detached NTP distribution at
        future positions + hard-target cross-entropy on assistant command tokens.
    """

    def __init__(
        self,
        base_model: nn.Module,
        hidden_size: int,
        horizon: int = 3,
        mtp_loss_weight: float = 0.30,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.base_model = base_model
        self.horizon = max(1, horizon)
        self.mtp_loss_weight = mtp_loss_weight
        self.temperature = temperature
        self.select_supervised_positions = (
            self.horizon == 1
            and getattr(getattr(base_model, "config", None), "model_type", None) == "llama"
            and "logits_to_keep" in inspect.signature(base_model.forward).parameters
        )

        # Lightweight residual projections for horizons k = 2..H (reuses lm_head)
        self.mtp_projs = nn.ModuleList(
            [
                nn.Sequential(
                    RMSNorm(hidden_size),
                    nn.Linear(hidden_size, hidden_size, bias=False),
                )
                for _ in range(max(0, self.horizon - 1))
            ]
        )

    def _get_lm_head(self) -> nn.Module:
        if hasattr(self.base_model, "lm_head"):
            return self.base_model.lm_head
        return self.base_model.get_output_embeddings()

    def forward_with_soft_mtp(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decisions: Optional[torch.Tensor] = None,
        contrast_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        shift_labels = labels[:, 1:]
        columns = None
        if self.select_supervised_positions:
            # Llama can project a shared list of sequence positions. Columns
            # masked in every row have zero loss and need no vocabulary logits.
            columns = torch.where(shift_labels.ne(-100).any(dim=0))[0]
            shift_labels = shift_labels.index_select(1, columns)
        # Handle both HuggingFace CausalLM and RecShellModel cleanly
        if isinstance(self.base_model, RecShellModel):
            out = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
            ntp_logits = out["logits"]
            hidden = out["last_hidden_state"]
        else:
            out = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=self.horizon > 1,
                use_cache=False,
                **({"logits_to_keep": columns} if columns is not None else {}),
            )
            ntp_logits = out.logits
            hidden = out.hidden_states[-1] if self.horizon > 1 else None

        # 1. Standard Next-Token Prediction (k = 1) cross-entropy
        shift_logits = ntp_logits if columns is not None else ntp_logits[:, :-1, :]
        valid = shift_labels != -100
        ntp_loss = F.cross_entropy(
            shift_logits[valid], shift_labels[valid],
        )

        loss = ntp_loss
        if decisions is not None and contrast_weight:
            # Both targets have the same prefix at this position. Its logits
            # already exist in the SFT pass, including the alternative token.
            rows = torch.where(decisions[:, 0] >= 0)[0]
            if rows.numel():
                positions, rejected = decisions[rows].to(input_ids.device).unbind(1)
                rows = rows.to(input_ids.device)
                chosen = labels[rows, positions + 1]
                if columns is not None:
                    positions = torch.searchsorted(columns, positions.contiguous())
                chosen_logits = ntp_logits[rows, positions, chosen]
                rejected_logits = ntp_logits[rows, positions, rejected]
                gap = rejected_logits.float() - chosen_logits.float()
                loss = loss + contrast_weight * F.softplus(gap).mean()

        if self.horizon <= 1 or self.mtp_loss_weight <= 0.0:
            return loss, {"ntp_loss": ntp_loss.item(), "mtp_loss": 0.0}

        # 2. Soft-MTP Self-Distillation for future horizons k = 2..H
        lm_head = self._get_lm_head()
        mtp_losses = []
        T = self.temperature

        for idx, proj in enumerate(self.mtp_projs):
            k = idx + 2  # predicting token at position t + k from hidden state at t
            if input_ids.size(1) <= k:
                continue

            h_future = hidden[:, :-k, :] + proj(hidden[:, :-k, :])
            pred_logits = lm_head(h_future)
            future_labels = labels[:, k:].contiguous()
            valid_mask = future_labels != -100

            if not valid_mask.any():
                continue

            # Hard CE component on valid assistant tokens
            hard_ce = F.cross_entropy(
                pred_logits.reshape(-1, pred_logits.size(-1)),
                future_labels.reshape(-1),
                ignore_index=-100,
            )

            # Soft-MTP KL self-distillation against detached NTP logits at t + k - 1
            # (teacher distribution predicting the exact same target token at t + k)
            teacher_logits = ntp_logits[:, k - 1 : -1, :].detach()
            p_student = F.log_softmax(pred_logits[valid_mask] / T, dim=-1)
            p_teacher = F.softmax(teacher_logits[valid_mask] / T, dim=-1)
            soft_kl = F.kl_div(p_student, p_teacher, reduction="batchmean") * (T * T)

            # Soft-MTP blend: 70% KL self-distillation + 30% hard label anchor
            mtp_losses.append(0.7 * soft_kl + 0.3 * hard_ce)

        if mtp_losses:
            mtp_loss = torch.stack(mtp_losses).mean()
            total_loss = loss + self.mtp_loss_weight * mtp_loss
            return total_loss, {"ntp_loss": ntp_loss.item(), "mtp_loss": mtp_loss.item()}

        return loss, {"ntp_loss": ntp_loss.item(), "mtp_loss": 0.0}


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
