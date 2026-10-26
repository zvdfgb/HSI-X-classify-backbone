"""Cross-modal messages at the last block's actual global/local stream outputs.

The original TransMixer classes are not modified. Extra modules are initialized
after the complete baseline, inside fork_rng, to preserve paired initialization.
"""
import torch
from torch import nn

from ..transmixer_net.model import TransMixerNet


VARIANTS = {
    "F0": "concat",
    "F1": "aux_self",
    "F2": "aux_to_hsi",
    "F3": "hsi_to_aux",
    "F4": "bidirectional",
    "F5": "split_matched",
    "F6": "split_swapped",
}
MODEL_NAMES = {key: f"TransMixer_Fusion_{key}" for key in VARIANTS}


class AttentionMessage(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        heads = next(h for h in (4, 2, 1) if dim % h == 0)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, target, source):
        # Weights are shared across directions, not separate model copies.
        q, kv = self.norm(target), self.norm(source)
        return self.attention(q, kv, kv, need_weights=False)[0]


class LocalMessage(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.source_dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.gate = nn.Conv2d(2 * dim, dim, 1)

    def forward(self, target, source, height, width):
        batch, _, dim = target.shape
        t = self.norm(target).transpose(1, 2).reshape(batch, dim, height, width)
        s = self.norm(source).transpose(1, 2).reshape(batch, dim, height, width)
        msg = self.source_dw(s) * torch.sigmoid(self.gate(torch.cat((t, s), 1)))
        return msg.flatten(2).transpose(1, 2)


class CrossModalInteraction(nn.Module):
    def __init__(self, dim, mode):
        super().__init__()
        if mode not in set(VARIANTS.values()):
            raise ValueError(f"Unknown fusion mode: {mode}")
        self.mode, self.dim = mode, dim
        self.strength = nn.ParameterDict()
        if mode == "concat":
            return
        if mode in ("split_matched", "split_swapped"):
            if dim % 2:
                raise ValueError("Split interaction requires an even hidden dimension")
            self.aux_in = nn.Linear(dim, dim)
            self.aux_out = nn.Linear(dim, dim)
            self.attention_message = AttentionMessage(dim // 2)
            self.local_message = LocalMessage(dim // 2)
            names = ("h_global", "h_local", "aux")
        else:
            self.attention_message = AttentionMessage(dim)
            names = {
                "aux_self": ("aux",), "aux_to_hsi": ("hsi",),
                "hsi_to_aux": ("aux",), "bidirectional": ("hsi", "aux"),
            }[mode]
        for name in names:
            self.strength[name] = nn.Parameter(torch.zeros(()))

    def forward(self, hsi, aux, height, width):
        if hsi.shape != aux.shape or hsi.shape[1] != height * width:
            raise ValueError("Interaction expects aligned B x (H*W) x D feature tensors")
        mode = self.mode
        if mode == "concat":
            return hsi, aux
        if mode == "aux_self":
            return hsi, aux + self.strength["aux"] * self.attention_message(aux, aux)
        if mode in ("aux_to_hsi", "hsi_to_aux", "bidirectional"):
            h_new, a_new = hsi, aux
            # Both messages use the original tensors: neither direction gets
            # privileged access to an already-updated other branch.
            if mode in ("aux_to_hsi", "bidirectional"):
                h_new = hsi + self.strength["hsi"] * self.attention_message(hsi, aux)
            if mode in ("hsi_to_aux", "bidirectional"):
                a_new = aux + self.strength["aux"] * self.attention_message(aux, hsi)
            return h_new, a_new
        hs = hsi.chunk(2, -1)
        parts = self.aux_in(aux).chunk(2, -1)
        h_messages, a_messages = [], []
        for i in range(2):
            use_attention = i == (0 if mode == "split_matched" else 1)
            if use_attention:
                h_messages.append(self.attention_message(hs[i], parts[i]))
                a_messages.append(self.attention_message(parts[i], hs[i]))
            else:
                h_messages.append(self.local_message(hs[i], parts[i], height, width))
                a_messages.append(self.local_message(parts[i], hs[i], height, width))
        h_new = torch.cat((
            hs[0] + self.strength["h_global"] * h_messages[0],
            hs[1] + self.strength["h_local"] * h_messages[1],
        ), -1)
        a_new = aux + self.strength["aux"] * self.aux_out(torch.cat(a_messages, -1))
        return h_new, a_new


class FusionTransMixerNet(TransMixerNet):
    def __init__(self, variant="F0", **kwargs):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown fusion variant: {variant}")
        if kwargs.pop("mode", "decoupled") != "decoupled":
            raise ValueError("Fusion ablation fixes the HSI backbone to decoupled B0")
        super().__init__(mode="decoupled", **kwargs)
        if not len(self.blocks):
            raise ValueError("At least one TransMixer block is required")
        self.variant = variant
        with torch.random.fork_rng(devices=[]):
            self.interaction = CrossModalInteraction(self.hidden_dim, VARIANTS[variant])

    def forward(self, hsi_pca, aux):
        if self.variant == "F0":
            return super().forward(hsi_pca, aux)
        if hsi_pca.dim() == 4:
            hsi_pca = hsi_pca.unsqueeze(1)
        batch, _, _, height, width = hsi_pca.shape
        h = self.hsi_stem(hsi_pca).reshape(batch, -1, height, width)
        tokens = self.hsi_proj(h).flatten(2).transpose(1, 2)
        for block in self.blocks[:-1]:
            tokens = block(tokens, height, width)
        block = self.blocks[-1]
        z, _ = block.ssm(block.norm1(tokens))
        g, l = z[:, :, :block.global_dim], z[:, :, block.global_dim:]
        # Preserve the original separate norm calls and MHA execution semantics.
        attn, _ = block.global_attn(block.global_norm(g), block.global_norm(g), block.global_norm(g))
        g = g + attn
        l = block.local_dw(l.transpose(1, 2).reshape(batch, block.local_dim, height, width))
        l = l.flatten(2).transpose(1, 2)
        a = self.aux_stem(aux).flatten(2).transpose(1, 2)
        streams, a = self.interaction(torch.cat((g, l), -1), a, height, width)
        tokens = tokens + block.fusion(streams)
        h = tokens.transpose(1, 2).reshape(batch, self.hidden_dim, height, width)
        a = a.transpose(1, 2).reshape(batch, self.hidden_dim, height, width)
        fused = self.fusion_conv(torch.cat((h, a), 1))
        return self.classifier(self.gap(fused).flatten(1))

    def interaction_strengths(self):
        return {name: float(value.detach().cpu()) for name, value in self.interaction.strength.items()}
