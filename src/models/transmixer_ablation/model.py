import math

import torch
from torch import nn
from ..transmixer_net.model import DecoupledTransMixerBlock, TransMixerNet


class MultiScaleDepthwise(nn.Module):
    """Preserve the original 3x3 path; add a zero-scaled 5x5 path before BN."""

    def __init__(self, original):
        super().__init__()
        self.dw3 = original
        channels = original.in_channels
        self.dw5 = nn.Conv2d(channels, channels, 5, padding=2,
                             groups=channels, bias=False)
        self.beta = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        return self.dw3(x) + self.beta * self.dw5(x)


class FeatureAblationBlock(DecoupledTransMixerBlock):
    def __init__(self, original, variant):
        super().__init__(dim=original.dim, split_ratio=0.5, mode="decoupled")
        # Copy all B0 parameters/buffers before introducing a single intervention.
        self.load_state_dict(original.state_dict(), strict=True)
        self.variant = variant
        if variant in ("B1", "B2"):
            self.alpha = nn.Parameter(torch.zeros(()))
        elif variant == "B3":
            # Only change dt bias. Preserve the baseline projection weights and A.
            # log-uniform targets, inverse softplus, following Mamba initialization.
            with torch.no_grad():
                dt = torch.exp(torch.empty(self.dim).uniform_(math.log(0.001), math.log(0.1)))
                self.ssm.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        elif variant == "B4":
            self.local_dw[0] = MultiScaleDepthwise(self.local_dw[0])
        else:
            raise ValueError(f"Unknown feature ablation: {variant}")

    def forward(self, x_tokens, H, W):
        if self.variant not in ("B1", "B2"):
            return super().forward(x_tokens, H, W)

        u = self.norm1(x_tokens)
        z, _ = self.ssm(u)
        x_g, x_l = z[:, :, :self.global_dim], z[:, :, self.global_dim:]
        # Match B0's separate Q/K/V norm calls (also preserves MHA fastpath choice).
        attn_out, _ = self.global_attn(
            self.global_norm(x_g), self.global_norm(x_g), self.global_norm(x_g)
        )
        global_out = x_g + attn_out
        batch, length, _ = x_l.shape
        local_out = self.local_dw(x_l.transpose(1, 2).reshape(batch, self.local_dim, H, W))
        local_out = local_out.reshape(batch, self.local_dim, length).transpose(1, 2)
        bypass = x_l if self.variant == "B1" else u[:, :, self.global_dim:]
        local_out = local_out + self.alpha * bypass
        return x_tokens + self.fusion(torch.cat([global_out, local_out], dim=-1))


class FeatureAblationNet(TransMixerNet):
    def __init__(self, variant, **kwargs):
        if variant not in ("B1", "B2", "B3", "B4"):
            raise ValueError(f"Unknown feature ablation: {variant}")
        super().__init__(mode="decoupled", **kwargs)
        self.variant = variant
        # Construct the complete B0 first: same seed -> identical shared weights,
        # including the classifier. Extra modules do not advance the caller's RNG.
        with torch.random.fork_rng(devices=[]):
            self.blocks = nn.ModuleList([
                FeatureAblationBlock(block, variant) for block in self.blocks
            ])
