"""Homogeneity-Aware Asymmetric Multi-Scale Fusion (HAMF) for HSI-LiDAR classification.

Native encoders, fusion_conv, SR2 readout, classifier, and training protocol are composed,
not copied or replaced. All 6 groups (M00-M05) use pure CrossEntropyLoss.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

GROUPS = tuple(f"M{i:02d}" for i in range(6))
PERMUTATION_SEED = 20261105


def fixed_derangement(length=121):
    """Deterministic spatial derangement using a private CPU generator."""
    gen = torch.Generator(device="cpu").manual_seed(PERMUTATION_SEED)
    ref = torch.arange(length)
    while True:
        perm = torch.randperm(length, generator=gen)
        if torch.all(perm != ref):
            return perm


class ScaledCrossModalRelations(nn.Module):
    """Asymmetric channel/spatial co-guidance with dot-product scaling and residual gates."""
    def __init__(self, seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 2026110510)
            self.value = nn.Conv2d(128, 128, 1)
            self.h_key = nn.Conv2d(64, 64, 1)
            self.l_key = nn.Conv2d(64, 64, 1)
        self.alpha_h = nn.Parameter(torch.tensor(0.1))
        self.alpha_l = nn.Parameter(torch.tensor(0.1))

    def forward(self, h, l):
        b, d, height, width = h.shape
        n = height * width
        vh, vl = self.value(torch.cat((h, l), 1)).flatten(2).chunk(2, 1)
        spatial = ((l.flatten(2).transpose(1, 2) @ self.h_key(h).flatten(2)) / math.sqrt(d)).softmax(-1)
        channel = ((h.flatten(2) @ self.l_key(l).flatten(2).transpose(1, 2)) / math.sqrt(n)).softmax(-1)
        h_out = h + self.alpha_h * (channel @ vh).reshape_as(h)
        l_out = l + self.alpha_l * (vl @ spatial).reshape_as(l)
        stats = {
            "alpha_h": float(self.alpha_h.detach()),
            "alpha_l": float(self.alpha_l.detach()),
            "spatial_attn_max": float(spatial.detach().max()),
            "channel_attn_max": float(channel.detach().max()),
        }
        return h_out, l_out, stats


def make_dw_pw_branch(k, seed_val):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed_val)
        return nn.Sequential(
            nn.Conv2d(128, 128, k, padding=k // 2, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.Dropout(0.1),
        )


class PixelScaleGate(nn.Module):
    """Pixel-wise 3-scale selector (1x1, 3x3, 5x5), optionally guided by SR homogeneity."""
    def __init__(self, group, seed):
        super().__init__()
        self.use_sr = group in ("M04", "M05")
        self.misaligned = group == "M05"
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 2026110540)
            self.feat_gate = nn.Sequential(
                nn.Conv2d(128, 16, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 3, 1),
            )
            nn.init.zeros_(self.feat_gate[-1].weight)
            nn.init.zeros_(self.feat_gate[-1].bias)
        if self.use_sr:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed + 2026110550)
                self.sr_gate = nn.Sequential(
                    nn.Conv2d(3, 8, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(8, 3, 1, bias=False),
                )
                nn.init.zeros_(self.sr_gate[-1].weight)
            self.register_buffer("sr_permutation", fixed_derangement(121))
        else:
            self.sr_gate = None

    def forward(self, summary_feat, sr_weights):
        logits = self.feat_gate(summary_feat)
        if self.use_sr:
            if sr_weights is None:
                raise RuntimeError("Native SR2 weights have not been prepared")
            w_sr = sr_weights.detach()
            if self.misaligned:
                b, _, h, w = w_sr.shape
                w_sr = w_sr.flatten(1)[:, self.sr_permutation].reshape(b, 1, h, w)
            s1 = 121.0 * w_sr
            s3 = F.avg_pool2d(s1, kernel_size=3, stride=1, padding=1, count_include_pad=False)
            s5 = F.avg_pool2d(s1, kernel_size=5, stride=1, padding=2, count_include_pad=False)
            logits = logits + self.sr_gate(torch.cat((s1, s3, s5), dim=1))
        return logits.softmax(dim=1)


class HAMFFusionStage(nn.Module):
    def __init__(self, original, pool, group, seed):
        super().__init__()
        if group not in GROUPS:
            raise ValueError(f"Unknown HAMF group: {group}")
        self.original = original
        self.group = group
        self.seed = int(seed)
        object.__setattr__(self, "_sr_pool", pool)  # Reference native SR pool without duplicate registration.
        self.disabled = False
        self.step = 0
        self.stats = {}
        self.history = []

        self.relations = None if group == "M00" else ScaledCrossModalRelations(self.seed)
        specs = (("b0_k3", 3, 30), ("b1_k3", 3, 3), ("b2_k3", 3, 31)) if group == "M01" else (
            ("b0_k1", 1, 1), ("b1_k3", 3, 3), ("b2_k5", 5, 5)
        )
        self.branches = nn.ModuleDict({
            name: make_dw_pw_branch(k, self.seed + 2026110520 + offset)
            for name, k, offset in specs
        })
        self.gamma = nn.Parameter(torch.full((1, 128, 1, 1), 0.1))
        self.scale_gate = PixelScaleGate(group, self.seed) if group in ("M03", "M04", "M05") else None

    @staticmethod
    def scale_entropy(w):
        w_safe = w.clamp_min(1e-8)
        return -(w_safe * w_safe.log()).sum(dim=1)

    def record_gradient(self, key, grad):
        name, value = key + "_gradient_norm", float(grad.norm())
        self.stats[name] = value
        if self.history and self.history[-1]["step"] == self.step:
            self.history[-1][name] = value

    def forward(self, x):
        if self.disabled:
            return self.original(x)
        h, l = x.chunk(2, 1)
        if self.training:
            for name, z in (("h", h), ("l", l)):
                if z.requires_grad:
                    z.register_hook(lambda grad, key=name: self.record_gradient(key, grad))

        rel_stats = {}
        if self.relations is not None:
            h, l, rel_stats = self.relations(h, l)
        x0 = torch.cat((h, l), 1)

        outs = [branch(x0) for branch in self.branches.values()]
        summary = (outs[0] + outs[1] + outs[2]) / 3.0
        if self.scale_gate is None:
            ms_out = summary
            w_stats = {
                "scale_w0_mean": 1.0 / 3.0,
                "scale_w1_mean": 1.0 / 3.0,
                "scale_w2_mean": 1.0 / 3.0,
                "scale_entropy_mean": math.log(3.0),
            }
        else:
            w = self.scale_gate(summary, self._sr_pool.weights)
            ms_out = w[:, 0:1] * outs[0] + w[:, 1:2] * outs[1] + w[:, 2:3] * outs[2]
            wd = w.detach()
            w_stats = {
                "scale_w0_mean": float(wd[:, 0].mean()),
                "scale_w1_mean": float(wd[:, 1].mean()),
                "scale_w2_mean": float(wd[:, 2].mean()),
                "scale_w0_Center": float(wd[:, 0, 5, 5].mean()),
                "scale_entropy_mean": float(self.scale_entropy(wd).mean()),
            }

        mixed = x0 + self.gamma * ms_out
        self.stats.update(
            **rel_stats,
            **w_stats,
            gamma_abs_mean=float(self.gamma.detach().abs().mean()),
            gamma_max=float(self.gamma.detach().max()),
        )
        if self.training:
            self.step += 1
        return self.original(mixed)

    def record_loss(self, ce):
        self.stats.update(step=self.step, ce=float(ce.detach()), auxiliary_loss=0.0)
        self.history.append(dict(self.stats))
        return ce
