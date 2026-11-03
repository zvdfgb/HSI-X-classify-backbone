"""Only add an input-derived key bias to the original MultiheadAttention.

No backbone block, attention computation, SR formula or classifier is copied.
Both controls have identical parameters; only the spatial prior correspondence
differs. The fixed permutation uses a private generator, never training RNG.
"""
import torch
from torch import nn

from ..transmixer_paper.model import NeighborTransMixer


PERMUTATION_SEED = 20261103
PRIOR_EPS = 1e-8


def fixed_derangement(length):
    generator = torch.Generator(device="cpu").manual_seed(PERMUTATION_SEED)
    reference = torch.arange(length)
    while True:
        permutation = torch.randperm(length, generator=generator)
        if torch.all(permutation != reference):
            return permutation


class SRPriorAttention(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original = original
        self.beta = nn.Parameter(torch.zeros(()))
        self.log_prior = None

    def forward(self, query, key, value, **kwargs):
        if self.log_prior is None:
            raise RuntimeError("SR prior must be prepared before native attention")
        batch, queries, _ = query.shape
        if self.log_prior.shape != (batch, key.shape[1]):
            raise ValueError("SR prior and attention spatial dimensions differ")
        if kwargs.get("attn_mask") is not None or kwargs.get("key_padding_mask") is not None:
            raise ValueError("This experiment only supports the original unmasked attention")
        # Every query receives the same input-derived prior over key positions.
        bias = self.beta * self.log_prior
        mask = bias[:, None, None, :].expand(batch, self.original.num_heads, queries, -1)
        kwargs["attn_mask"] = mask.reshape(batch * self.original.num_heads, queries, -1)
        return self.original(query, key, value, **kwargs)


class SRPriorTransMixer(NeighborTransMixer):
    def __init__(self, prior_mode, **kwargs):
        if prior_mode not in ("aligned", "misaligned"):
            raise ValueError(prior_mode)
        super().__init__("SR2", **kwargs)
        if self.mode != "decoupled":
            raise ValueError("Keep native fixed 1:1 local/global channel assignment")
        self.prior_mode = prior_mode
        self.register_buffer("prior_permutation", fixed_derangement(self.window_size ** 2))
        # Construct the complete native model first, then wrap its actual MHA.
        # Scalar zeros and private permutation generation consume no global RNG.
        for block in self.blocks:
            block.global_attn = SRPriorAttention(block.global_attn)

    def neighbor_weights(self, hsi, aux):
        weights = super().neighbor_weights(hsi, aux)
        flat = weights.flatten(1)
        prior = flat if self.prior_mode == "aligned" else flat[:, self.prior_permutation]
        log_prior = (prior + PRIOR_EPS).log().detach()
        for block in self.blocks:
            block.global_attn.log_prior = log_prior
        # Both groups keep the ORIGINAL, correctly aligned SR pooling weights.
        return weights

    def diagnostics(self):
        return {"prior_mode": self.prior_mode,
                "beta_by_block": [float(b.global_attn.beta.detach().cpu()) for b in self.blocks],
                "permutation": self.prior_permutation.detach().cpu().tolist(),
                "prior_eps": PRIOR_EPS, "permutation_seed": PERMUTATION_SEED}
