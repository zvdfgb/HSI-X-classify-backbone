"""Boundary-Aware Multi-Scale Asymmetric Fusion (BMAF) for HSI-LiDAR classification.

Native encoders, fusion_conv, SR2 readout, classifier, and training protocol are composed,
not copied or replaced. All 6 groups (N00-N05) use pure CrossEntropyLoss.
"""
import torch
from torch import nn
from torch.nn import functional as F

GROUPS = tuple(f"N{i:02d}" for i in range(6))
PERMUTATION_SEED = 20261106


def fixed_derangement(length=121):
    """Deterministic spatial derangement using a private CPU generator."""
    gen = torch.Generator(device="cpu").manual_seed(PERMUTATION_SEED)
    ref = torch.arange(length)
    while True:
        perm = torch.randperm(length, generator=gen)
        if torch.all(perm != ref):
            return perm


def _cache_raw_inputs_pre_hook(module, args):
    """Capture detached raw (hsi_pca, aux) for local boundary variance without modifying NeighborTransMixer."""
    if len(args) >= 2 and hasattr(module, "fusion_conv") and hasattr(module.fusion_conv, "set_raw_inputs"):
        module.fusion_conv.set_raw_inputs(args[0], args[1])


class CrossModalRelations(nn.Module):
    """Un-damped asymmetric channel/spatial co-guidance (exact C02 formulation)."""
    def __init__(self, seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.value = nn.Conv2d(128, 128, 1)
            self.h_key = nn.Conv2d(64, 64, 1)
            self.l_key = nn.Conv2d(64, 64, 1)

    def forward(self, h, l):
        vh, vl = self.value(torch.cat((h, l), 1)).flatten(2).chunk(2, 1)
        spatial = (l.flatten(2).transpose(1, 2) @ self.h_key(h).flatten(2)).softmax(-1)
        channel = (h.flatten(2) @ self.l_key(l).flatten(2).transpose(1, 2)).softmax(-1)
        h_out = h + (channel @ vh).reshape_as(h)
        l_out = l + (vl @ spatial).reshape_as(l)
        sd = spatial.detach()
        cd = channel.detach()
        sp_safe = sd.clamp_min(1e-8)
        ch_safe = cd.clamp_min(1e-8)
        stats = {
            "spatial_attn_max": float(sd.max()),
            "channel_attn_max": float(cd.max()),
            "spatial_attn_entropy": float((-(sp_safe * sp_safe.log()).sum(-1)).mean()),
            "channel_attn_entropy": float((-(ch_safe * ch_safe.log()).sum(-1)).mean()),
        }
        return h_out, l_out, stats


def make_sharp_1x1_branch(out_ch, seed_val):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed_val)
        return nn.Sequential(
            nn.Conv2d(128, out_ch, 1),
            nn.ReLU(),
            nn.BatchNorm2d(out_ch),
            nn.Dropout(0.1),
        )


def make_bottleneck_dense_branch(out_ch, k, seed_val):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed_val)
        return nn.Sequential(
            nn.Conv2d(128, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, k, padding=k // 2),
            nn.ReLU(),
            nn.BatchNorm2d(out_ch),
            nn.Dropout(0.1),
        )


def _local_norm_var(z, k):
    pad = k // 2
    z_c = z - z.mean(dim=(2, 3), keepdim=True)
    z_pad = F.pad(z_c, (pad, pad, pad, pad), mode="replicate")
    mu = F.avg_pool2d(z_pad, kernel_size=k, stride=1)
    var = (F.avg_pool2d(z_pad.square(), kernel_size=k, stride=1) - mu.square()).clamp_min(0.0)
    var_mean = var.mean(dim=1, keepdim=True)
    return var_mean / (var_mean.mean(dim=(2, 3), keepdim=True) + 1e-5)


def compute_boundary_map(hsi_pca, aux, dtype=torch.float32):
    with torch.no_grad():
        h = (hsi_pca[:, 0] if hsi_pca.ndim == 5 else hsi_pca).detach().to(dtype=dtype)
        l = (aux[:, 0] if aux.ndim == 5 else aux).detach().to(dtype=dtype)
        vh3 = _local_norm_var(h, 3)
        vl3 = _local_norm_var(l, 3)
        vh5 = _local_norm_var(h, 5)
        vl5 = _local_norm_var(l, 5)
        return torch.cat((vh3, vl3, vh5, vl5), dim=1)


class BoundaryScaleModulator(nn.Module):
    """Bounded pixel-wise modulation g3, g5 in (0.5, 1.5) (exact 1.0 at step 0), keeping 1x1 branch at 1.0."""
    def __init__(self, group, seed):
        super().__init__()
        self.use_bnd = group in ("N04", "N05")
        self.misaligned = group == "N05"
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 2026110640)
            self.feat_gate = nn.Sequential(
                nn.Conv2d(128, 16, 1),
                nn.ReLU(),
                nn.Conv2d(16, 2, 1),
            )
            nn.init.zeros_(self.feat_gate[-1].weight)
            nn.init.zeros_(self.feat_gate[-1].bias)
        if self.use_bnd:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed + 2026110650)
                self.bnd_gate = nn.Sequential(
                    nn.Conv2d(4, 8, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(8, 2, 1, bias=False),
                )
                nn.init.zeros_(self.bnd_gate[-1].weight)
            self.register_buffer("bnd_permutation", fixed_derangement(121))
        else:
            self.bnd_gate = None

    def forward(self, u, raw_inputs):
        logits = self.feat_gate(u)
        if self.use_bnd:
            if raw_inputs is None:
                raise RuntimeError("Raw multimodal inputs (hsi_pca, aux) were not prepared")
            bnd = compute_boundary_map(raw_inputs[0], raw_inputs[1], dtype=u.dtype)
            if self.misaligned:
                b, c, height, width = bnd.shape
                bnd = bnd.flatten(2)[:, :, self.bnd_permutation].reshape(b, c, height, width)
            logits = logits + self.bnd_gate(bnd)
        return 1.0 + 0.5 * torch.tanh(logits)


class BMAFFusionStage(nn.Module):
    def __init__(self, original, pool, group, seed):
        super().__init__()
        if group not in GROUPS:
            raise ValueError(f"Unknown BMAF group: {group}")
        self.original = original
        self.group = group
        self.seed = int(seed)
        object.__setattr__(self, "_sr_pool", pool)  # Reference native SR pool without duplicate registration.
        self._raw_inputs = None
        self.disabled = False
        self.step = 0
        self.stats = {}
        self.history = []

        self.relations = None if group == "N00" else CrossModalRelations(self.seed)
        if group == "N01":
            self.branches = nn.ModuleDict({
                "b1": make_bottleneck_dense_branch(44, 3, self.seed + 2026110631),
                "b3": make_bottleneck_dense_branch(42, 3, self.seed + 2026110623),
                "b5": make_bottleneck_dense_branch(42, 3, self.seed + 2026110635),
            })
        else:
            self.branches = nn.ModuleDict({
                "b1": make_sharp_1x1_branch(44, self.seed + 2026110621),
                "b3": make_bottleneck_dense_branch(42, 3, self.seed + 2026110623),
                "b5": make_bottleneck_dense_branch(42, 5, self.seed + 2026110625),
            })
        self.modulator = BoundaryScaleModulator(group, self.seed) if group in ("N03", "N04", "N05") else None

    def set_raw_inputs(self, hsi_pca, aux):
        self._raw_inputs = (hsi_pca.detach(), aux.detach())

    def record_gradient(self, key, grad):
        name, value = key + "_gradient_norm", float(grad.norm())
        self.stats[name] = value
        if self.history and self.history[-1]["step"] == self.step:
            self.history[-1][name] = value

    def forward(self, x):
        raw_inputs, self._raw_inputs = self._raw_inputs, None
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

        y1 = self.branches["b1"](x0)
        y3 = self.branches["b3"](x0)
        y5 = self.branches["b5"](x0)
        u = torch.cat((y1, y3, y5), dim=1)

        if self.modulator is None:
            mixed = u
            g_stats = {
                "g3_mean": 1.0,
                "g5_mean": 1.0,
                "g3_min": 1.0,
                "g3_max": 1.0,
                "g5_min": 1.0,
                "g5_max": 1.0,
                "g3_center": 1.0,
                "g5_center": 1.0,
            }
        else:
            g = self.modulator(u, raw_inputs)
            g3, g5 = g[:, 0:1], g[:, 1:2]
            mixed = torch.cat((y1, g3 * y3, g5 * y5), dim=1)
            gd = g.detach()
            g_stats = {
                "g3_mean": float(gd[:, 0].mean()),
                "g5_mean": float(gd[:, 1].mean()),
                "g3_min": float(gd[:, 0].min()),
                "g3_max": float(gd[:, 0].max()),
                "g5_min": float(gd[:, 1].min()),
                "g5_max": float(gd[:, 1].max()),
                "g3_center": float(gd[:, 0, 5, 5].mean()),
                "g5_center": float(gd[:, 1, 5, 5].mean()),
            }

        self.stats.update(**rel_stats, **g_stats)
        if self.training:
            self.step += 1
        return self.original(mixed)

    def record_loss(self, ce):
        self.stats.update(step=self.step, ce=float(ce.detach()), auxiliary_loss=0.0)
        self.history.append(dict(self.stats))
        return ce
