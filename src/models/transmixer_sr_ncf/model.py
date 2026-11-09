"""Compact shared-kernel multi-scale and native SR-guided cross-modal neighborhood fusion (SR-NCF).

Native encoders, fusion_conv, SR2/GAP readout, classifier, and training protocol are composed,
not copied or replaced. All 7 groups (K00-K06) use pure CrossEntropyLoss.
"""
import torch
from torch import nn
from torch.nn import functional as F

GROUPS = tuple(f"K{i:02d}" for i in range(7))
PERMUTATION_SEED = 20261107


def fixed_derangement(length=121):
    """Deterministic spatial derangement using a private CPU generator (zero training RNG consumption)."""
    gen = torch.Generator(device="cpu").manual_seed(PERMUTATION_SEED)
    ref = torch.arange(length, dtype=torch.long)
    while True:
        perm = torch.randperm(length, generator=gen)
        if torch.all(perm != ref):
            return perm


def local_mean_one_tap_weights(w_map, dilation, in_bounds_mask):
    """Normalize patch-center-referenced SR2 weights over valid in-window neighbors to exact mean 1.0.

    Args:
        w_map: [B, 1, H, W] non-negative spatial prior over the 11x11 window.
        dilation: sampling interval d in {1, 2} (zero-padding padding=d).
        in_bounds_mask: [1, 9, H*W] binary indicator of valid in-window neighbors (1 inside, 0 padded).

    Returns:
        alpha: [B, 1, 9, H*W] modulation weights multiplied onto unfolded neighbor contributions
               BEFORE cross-channel shared 3x3 convolution. Satisfies:
               1) alpha == 0 on out-of-window zero-padded positions;
               2) arithmetic mean over valid in-window neighbors equals 1.0 at every output position
                  (normalized in float64 down to float32 normal limit 1e-37 so small corner SR2 weights
                  around 1e-32..1e-13 are never zeroed by premature 1e-12 clamping; if all valid neighbors
                  underflow float32 softmax <= 1e-37, falls back cleanly to uniform valid weight 1.0);
               3) under uniform prior (w_map == const > 0), alpha == 1.0 on all valid neighbors,
                  degenerating identically to standard zero-padded F.conv2d(..., padding=d, dilation=d).
    """
    w64 = w_map.to(torch.float64)
    m64 = in_bounds_mask.to(torch.float64)
    n_valid = m64.sum(dim=1, keepdim=True)  # [1, 1, H*W], values in {4, 6, 9}
    w_unf = F.unfold(w64, kernel_size=3, dilation=dilation, padding=dilation)  # [B, 9, H*W]
    w_valid = w_unf * m64
    denom = w_valid.sum(dim=1, keepdim=True)  # [B, 1, H*W]
    alpha64 = torch.where(
        denom > 1e-37,
        m64 * n_valid * w_valid / denom.clamp_min(1e-300),
        m64,
    )
    return alpha64.to(dtype=w_map.dtype).unsqueeze(1)  # [B, 1, 9, H*W]


def _make_conv1x1_bn_relu_drop(in_ch, out_ch, seed_val, drop_p=0.1):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed_val)
        layers = [
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if drop_p > 0.0:
            layers.append(nn.Dropout(drop_p))
        return nn.Sequential(*layers)


def _make_bottleneck_unshared_branch(out_ch, kernel_size, seed_val):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed_val)
        return nn.Sequential(
            nn.Conv2d(128, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )


class SRNCFFusionStage(nn.Module):
    def __init__(self, original, pool, group, seed, window_size=11):
        super().__init__()
        if group not in GROUPS:
            raise ValueError(f"Unknown SR-NCF group: {group}")
        self.original = original
        self.group = group
        self.seed = int(seed)
        object.__setattr__(self, "_sr_pool", pool)  # Reference native SR pool without duplicate registration.
        self.disabled = False
        self.step = 0
        self.stats = {}
        self.history = []

        ones = torch.ones((1, 1, window_size, window_size), dtype=torch.float32)
        self.register_buffer("mask_d1", F.unfold(ones, kernel_size=3, dilation=1, padding=1), persistent=False)
        self.register_buffer("mask_d2", F.unfold(ones, kernel_size=3, dilation=2, padding=2), persistent=False)
        self.register_buffer("fusion_permutation", fixed_derangement(window_size * window_size), persistent=False)

        if group == "K00":
            # Matched-budget (~108.6k) ordinary unshared 1x1(92) + 3x3(18) + 5x5(18) control
            self.b0 = _make_conv1x1_bn_relu_drop(128, 92, self.seed + 2026110710, drop_p=0.1)
            self.b3 = _make_bottleneck_unshared_branch(18, 3, self.seed + 2026110713)
            self.b5 = _make_bottleneck_unshared_branch(18, 5, self.seed + 2026110715)
        else:
            # K01..K06 share bit-identical architecture, parameter count (108,111), and initial weights!
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(self.seed + 2026110720)
                self.direct_1x1 = nn.Sequential(
                    nn.Conv2d(128, 44, 1, bias=False),
                    nn.BatchNorm2d(44),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1),
                )
                self.compress_1x1 = nn.Sequential(
                    nn.Conv2d(128, 42, 1, bias=False),
                    nn.BatchNorm2d(42),
                    nn.ReLU(inplace=True),
                )
                self.shared_conv3x3 = nn.Conv2d(42, 42, 3, padding=1, bias=False)
                self.bn_d1 = nn.BatchNorm2d(42)
                self.bn_d2 = nn.BatchNorm2d(42)
                self.act = nn.ReLU(inplace=True)
                self.drop = nn.Dropout(0.1)

    def record_gradient(self, key, grad):
        name, value = key + "_gradient_norm", float(grad.norm())
        self.stats[name] = value
        if self.history and self.history[-1]["step"] == self.step:
            self.history[-1][name] = value

    def _shared_atrous_branch(self, z, dilation, mask, bn, w_fusion):
        b, c, h, w = z.shape
        alpha = local_mean_one_tap_weights(w_fusion, dilation, mask)  # [B, 1, 9, H*W]
        z_unf = F.unfold(z, kernel_size=3, dilation=dilation, padding=dilation).view(b, c, 9, h * w)
        z_mod = (z_unf * alpha).view(b, c * 9, h * w)
        w_mat = self.shared_conv3x3.weight.view(c, c * 9)
        conv_out = (w_mat @ z_mod).view(b, c, h, w)
        ad = alpha.detach()
        valid_sum = ad.squeeze(1).sum(dim=1)  # [B, H*W]
        n_val = mask.sum(dim=1)  # [1, H*W]
        center_idx = (h // 2) * w + (w // 2)
        b_stats = {
            f"alpha_d{dilation}_valid_mean": float((valid_sum / n_val).mean()),
            f"alpha_d{dilation}_max": float(ad.max()),
            f"alpha_d{dilation}_center_self": float(ad[:, 0, 4, center_idx].mean()),
        }
        return self.drop(self.act(bn(conv_out))), b_stats

    def forward(self, x):
        if self.disabled:
            return self.original(x)

        w_sr = self._sr_pool.weights
        if w_sr is None or w_sr.shape[-2:] != x.shape[-2:]:
            raise RuntimeError("Native SR2 weights were not prepared before fusion_conv")

        h_feat, l_feat = x.chunk(2, 1)
        if self.training:
            for name, z_mod in (("h", h_feat), ("l", l_feat)):
                if z_mod.requires_grad:
                    z_mod.register_hook(lambda grad, key=name: self.record_gradient(key, grad))
        x_in = torch.cat((h_feat, l_feat), dim=1)

        if self.group == "K00":
            u = torch.cat((self.b0(x_in), self.b3(x_in), self.b5(x_in)), dim=1)
            w_ro = self._sr_pool.weights.detach().clamp_min(1e-12)
            ro_ent = float((-(w_ro * w_ro.log()).sum(dim=(2, 3))).mean())
            self.stats.update(
                alpha_d1_valid_mean=1.0,
                alpha_d2_valid_mean=1.0,
                alpha_d1_max=1.0,
                alpha_d2_max=1.0,
                alpha_d1_center_self=1.0,
                alpha_d2_center_self=1.0,
                readout_entropy=ro_ent,
            )
            if self.training:
                self.step += 1
            return self.original(u)

        # 1. Select fusion-stage neighborhood prior (before touching readout weights):
        num_px = float(w_sr.shape[-2] * w_sr.shape[-1])
        if self.group in ("K01", "K06"):
            # Uniform prior executed through the exact same unfold+matmul path (zero CUDA kernel confound!)
            w_fusion = torch.full_like(w_sr.detach(), 1.0 / num_px)
        elif self.group in ("K02", "K04", "K05"):
            w_fusion = w_sr.detach()
        elif self.group == "K03":
            # Misalign ONLY the fusion prior; readout self._sr_pool.weights stays intact true SR2!
            bsz, ch, height, width = w_sr.shape
            w_fusion = w_sr.detach().flatten(2)[:, :, self.fusion_permutation].reshape(bsz, ch, height, width)
        else:
            raise RuntimeError(f"Unhandled group: {self.group}")

        # 2. Select readout-stage pooling weights (K04 & K06 replace SR2 with uniform GAP 1/121):
        if self.group in ("K04", "K06"):
            self._sr_pool.weights = torch.full_like(w_sr, 1.0 / num_px)

        y0 = self.direct_1x1(x_in)
        z = self.compress_1x1(x_in)
        y1, st1 = self._shared_atrous_branch(z, 1, self.mask_d1, self.bn_d1, w_fusion)
        y2, st2 = self._shared_atrous_branch(z, 2, self.mask_d2, self.bn_d2, w_fusion)
        u = torch.cat((y0, y1, y2), dim=1)
        if self.group == "K05":
            u = x_in + u

        w_ro = self._sr_pool.weights.detach().clamp_min(1e-12)
        ro_ent = float((-(w_ro * w_ro.log()).sum(dim=(2, 3))).mean())
        self.stats.update(**st1, **st2, readout_entropy=ro_ent)
        if self.training:
            self.step += 1
        return self.original(u)

    def record_loss(self, ce):
        self.stats.update(step=self.step, ce=float(ce.detach()), auxiliary_loss=0.0)
        self.history.append(dict(self.stats))
        return ce
