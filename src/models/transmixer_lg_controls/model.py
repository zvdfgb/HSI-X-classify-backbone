"""Replace one branch operator while reusing the original block forward.

The two original branch wrappers are intentionally retained: the first has
LayerNorm + residual, the second does not. These are operator controls, not a
claim of identical capacity or proof that a 1:1 ratio is optimal.
"""
import torch
from torch import nn


class LocalOperatorAsAttention(nn.Module):
    def __init__(self, dim, window):
        super().__init__()
        self.window = window
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim), nn.GELU(),
            nn.Conv2d(dim, dim, 1, bias=False), nn.BatchNorm2d(dim))

    def forward(self, query, key, value, **kwargs):
        b, length, dim = query.shape
        assert length == self.window**2
        output = self.net(query.transpose(1, 2).reshape(b, dim, self.window, self.window))
        return output.flatten(2).transpose(1, 2), None


class GlobalOperatorAsConvolution(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, 4, batch_first=True)

    def forward(self, x):
        b, dim, height, width = x.shape
        tokens = self.norm(x.flatten(2).transpose(1, 2))
        output = self.attention(tokens, tokens, tokens)[0]
        return output.transpose(1, 2).reshape(b, dim, height, width)


def replace_operators(net, kind):
    if kind not in ('LL', 'GG'):
        raise ValueError(kind)
    # Preserve the base network's initialization and global RNG state.
    with torch.random.fork_rng(devices=[]):
        for block in net.blocks:
            assert block.mode == 'decoupled' and block.global_dim == block.local_dim == 32
            if kind == 'LL':
                block.global_attn = LocalOperatorAsAttention(32, net.window_size)
            else:
                block.local_dw = GlobalOperatorAsConvolution(32)
    net.operator_control = kind
    return net
