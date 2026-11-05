"""Candidate only: native SSM/operators/readout with balanced branch updates.

Fixed 32/32 channels. Input-derived SR informs sample-level update gains;
it never changes channel assignment or the original final SR2 formula.
"""
import torch
from torch import nn

from ..transmixer_paper.model import NeighborTransMixer


class BalancedSRBlock(nn.Module):
    def __init__(self, original, summary_mode):
        super().__init__()
        if summary_mode not in ("uniform", "aligned", "misaligned"):
            raise ValueError("Unknown SR summary mode")
        if original.mode != "decoupled" or original.global_dim != original.local_dim:
            raise ValueError("Require the original fixed 1:1 decoupled block")
        self.dim, self.mode = original.dim, original.mode
        self.global_dim, self.local_dim = original.global_dim, original.local_dim
        self.summary_mode = summary_mode
        # Reuse the actual original layers, not copied implementations.
        for name in ("norm1", "ssm", "global_attn", "global_norm", "local_dw", "fusion"):
            setattr(self, name, getattr(original, name))
        self.local_norm = nn.LayerNorm(self.local_dim)
        self.global_update_norm = nn.LayerNorm(self.global_dim)
        self.local_update_norm = nn.LayerNorm(self.local_dim)
        self.branch_gate = nn.Sequential(nn.Linear(self.dim, 16), nn.GELU(), nn.Linear(16, 2))
        nn.init.zeros_(self.branch_gate[-1].weight)
        nn.init.zeros_(self.branch_gate[-1].bias)
        self.summary_weights = None
        self.last_gains = None

    def summarize(self, global_update, local_update):
        updates = torch.cat((global_update, local_update), dim=-1)
        if self.summary_mode == "uniform":
            return updates.mean(dim=1)
        if self.summary_weights is None or self.summary_weights.shape != updates.shape[:2]:
            raise RuntimeError("Fresh input SR summary weights required")
        return (updates * self.summary_weights.unsqueeze(-1)).sum(dim=1)

    def forward(self, x_tokens, height, width):
        if height != 11 or width != 11 or x_tokens.shape[1] != 121:
            raise ValueError("Candidate is scoped to the approved 11x11 window")
        z, _ = self.ssm(self.norm1(x_tokens))
        global_x, local_x = z.split((self.global_dim, self.local_dim), dim=-1)
        # Same native Q/K/V normalization calls as the historical block.
        global_delta, _ = self.global_attn(self.global_norm(global_x),
            self.global_norm(global_x), self.global_norm(global_x))
        batch, length, _ = local_x.shape
        local_input = self.local_norm(local_x).transpose(1, 2).reshape(batch, self.local_dim, height, width)
        local_delta = self.local_dw(local_input).flatten(2).transpose(1, 2)
        global_delta = self.global_update_norm(global_delta)
        local_delta = self.local_update_norm(local_delta)
        gains = 2 * torch.sigmoid(self.branch_gate(self.summarize(global_delta, local_delta)))
        self.last_gains = gains.detach()
        global_out = global_x + gains[:, 0, None, None] * global_delta
        local_out = local_x + gains[:, 1, None, None] * local_delta
        return x_tokens + self.fusion(torch.cat((global_out, local_out), dim=-1))


class SRBalancedTransMixer(NeighborTransMixer):
    def __init__(self, summary_mode, **kwargs):
        if summary_mode not in ("uniform", "aligned", "misaligned"):
            raise ValueError("Unknown SR summary mode")
        if kwargs.get("mode", "decoupled") != "decoupled":
            raise ValueError("Retain fixed 1:1 decoupled assignment")
        if kwargs.get("window_size", 11) != 11 or kwargs.get("hidden_dim", 64) != 64:
            raise ValueError("Retain hidden64 and window11")
        super().__init__("SR2", **kwargs)
        self.summary_mode = summary_mode
        self.register_buffer("summary_permutation", torch.roll(torch.arange(121), shifts=37))
        # The original complete backbone/classifier initializes before any additions.
        with torch.random.fork_rng(devices=[]):
            self.blocks = nn.ModuleList([BalancedSRBlock(b, summary_mode) for b in self.blocks])

    def neighbor_weights(self, hsi, aux):
        weights = super().neighbor_weights(hsi, aux)
        summary = weights.flatten(1)
        if self.summary_mode == "misaligned":
            summary = summary[:, self.summary_permutation]
        for block in self.blocks:
            block.summary_weights = summary.detach()
        # Final pooling always uses the original, correctly aligned SR weights.
        return weights

    def diagnostics(self):
        return {"summary_mode": self.summary_mode, "split": [32, 32],
            "gain_range": [0, 2], "gains_last_batch": [
                None if b.last_gains is None else b.last_gains.cpu().tolist() for b in self.blocks]}
