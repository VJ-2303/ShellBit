import json
import tempfile
import unittest
from pathlib import Path

import torch

from model_arch import RecShellConfig, RecShellModel, SoftMTPWrapper
from train_sft import BashSFTDataset, collate_commands


class BytesTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=True):
        return list(text.encode())


class ContrastTests(unittest.TestCase):
    def test_alignment_padding_and_loss(self):
        rows = [{"prompt": "p", "target": "sort -n", "rejected_target": "sort -r"},
                {"prompt": "abc", "target": "ls"},
                {"prompt": "x", "target": "x" * 32}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "train.jsonl")
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            ds = BashSFTDataset(path, BytesTokenizer(), 32)
        self.assertEqual(len(ds), 2)
        batch = collate_commands(ds.samples, 0)
        ids, labels = batch["input_ids"], batch["labels"]
        self.assertEqual(tuple(ids.shape), (2, 9))
        self.assertEqual(batch["decisions"].tolist(), [[6, ord("r")], [-1, -1]])
        self.assertEqual(labels[1].tolist(), [-100, -100, -100, 108, 115, 0, -100, -100, -100])
        cfg = RecShellConfig(vocab_size=128, d_model=16, n_heads=2, n_kv_heads=1,
                             n_recursions=1, n_physical_layers=1, ffn_dim=32, use_bitnet_158=False)
        model = RecShellModel(cfg)
        wrapper = SoftMTPWrapper(model, 16, horizon=1)
        baseline, _ = wrapper.forward_with_soft_mtp(ids, labels)
        combined, _ = wrapper.forward_with_soft_mtp(ids, labels, decisions=batch["decisions"], contrast_weight=.25)
        raw = model(ids)["logits"][0, 6]
        expected = torch.nn.functional.softplus(raw[ord("r")] - raw[ord("n")])
        torch.testing.assert_close(combined, baseline + .25 * expected)
        combined.backward()
        self.assertTrue(torch.isfinite(model.lm_head.weight.grad).all())

    def test_selective_projection_matches_dense_loss_and_gradient(self):
        from types import SimpleNamespace

        class Decoder(torch.nn.Module):
            def __init__(self, kind):
                super().__init__()
                self.config = SimpleNamespace(model_type=kind)
                self.emb = torch.nn.Embedding(32, 8)
                self.lm_head = torch.nn.Linear(8, 32)

            def forward(self, input_ids, logits_to_keep=None, **kwargs):
                hidden = self.emb(input_ids)
                if logits_to_keep is not None:
                    hidden = hidden.index_select(1, logits_to_keep)
                self.projected = hidden.size(1)
                return SimpleNamespace(logits=self.lm_head(hidden))

        dense = Decoder("generic")
        selected = Decoder("llama")
        selected.load_state_dict(dense.state_dict())
        ids = torch.tensor([[1, 2, 3, 4, 5, 6], [10, 12, 13, 14, 15, 16]])
        labels = torch.tensor([[-100, -100, -100, -100, 5, 6], [-100, -100, 13, -100, -100, -100]])
        losses = []
        for model in (dense, selected):
            wrapper = SoftMTPWrapper(model, 8, horizon=1)
            loss, _ = wrapper.forward_with_soft_mtp(ids, labels, decisions=torch.tensor([[3, 10], [1, 20]]),
                                                    contrast_weight=0.2)
            loss.backward()
            losses.append(loss)
        torch.testing.assert_close(losses[0], losses[1])
        torch.testing.assert_close(dense.emb.weight.grad, selected.emb.weight.grad)
        torch.testing.assert_close(dense.lm_head.weight.grad, selected.lm_head.weight.grad)
        self.assertEqual((dense.projected, selected.projected), (6, 3))
