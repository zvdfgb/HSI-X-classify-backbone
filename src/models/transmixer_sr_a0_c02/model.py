"""Native SR2-guided neighborhood fusion applied cleanly to A0 (R00-R02) and C02 (R03-R05).

Native encoders, CFTransfer('C02'), fusion_conv, SR2 readout, classifier, and training protocol
are composed without duplication or replacement. All 6 groups use pure CrossEntropyLoss.
"""
import torch
from torch import nn
from torch.nn import functional as F

from ..transmixer_fusion_v3.model import CFTransfer
from ..transmixer_sr_ncf.model import fixed_derangement

GROUPS = tuple(f"R{i:02d}" for i in range(6))


def local_mean_one_kernel_weights(w_map, kernel_size, in_bounds_mask):
    """Normalize patch-center-referenced SR2 weights over valid in-window k*k neighbors to exact mean 1.0.

    Args:
        w_map: [B, 1, H, W] non-negative spatial prior over the 11x11 window.
        kernel_size: odd spatial kernel size k in {3, 5} (zero-padding padding = k // 2).
        in_bounds_mask: [1, k*k, H*W] binary indicator of valid in-window neighbors (1 inside, 0 padded).

    Returns:
        alpha: [B, 1, k*k, H*W] modulation weights multiplied onto unfolded neighbor contributions
               BEFORE cross-channel convolution. Satisfies:
               1) alpha == 0 on out-of-window zero-padded positions;
               2) arithmetic mean over valid in-window neighbors equals 1.0 at every output position
                  (normalized in float64 down to float32 normal limit 1e-37; falls back cleanly to
                  uniform valid weight 1.0 only if all valid neighbors underflow <= 1e-37);
               3) under uniform prior (w_map == const > 0), alpha == 1.0 on all valid neighbors,
                  degenerating identically to standard zero-padded F.conv2d(..., padding=k//2).
    """
    pad = kernel_size // 2
    w64 = w_map.to(torch.float64)
    m64 = in_bounds_mask.to(torch.float64)
    n_valid = m64.sum(dim=1, keepdim=True)  # [1, 1, H*W]
    w_unf = F.unfold(w64, kernel_size=kernel_size, padding=pad)  # [B, k*k, H*W]
    w_valid = w_unf * m64
    denom = w_valid.sum(dim=1, keepdim=True)  # [B, 1, H*W]
    alpha64 = torch.where(
        denom > 1e-37,
        m64 * n_valid * w_valid / denom.clamp_min(1e-300),
        m64,
    )
    return alpha64.to(dtype=w_map.dtype).unsqueeze(1)  # [B, 1, k*k, H*W]


def sr_modulated_conv2d(x, conv_layer, in_bounds_mask, w_fusion):
    """Apply cross-channel Conv2d with per-neighbor local-mean-1 modulation before convolution."""
    b, c_in, h, w = x.shape
    c_out = conv_layer.out_channels
    k = conv_layer.kernel_size[0]
    alpha = local_mean_one_kernel_weights(w_fusion, k, in_bounds_mask)  # [B, 1, k*k, H*W]
    x_unf = F.unfold(x, kernel_size=k, padding=k // 2).view(b, c_in, k * k, h * w)
    x_mod = (x_unf * alpha).view(b, c_in * k * k, h * w)
    out = (conv_layer.weight.view(c_out, c_in * k * k) @ x_mod).view(b, c_out, h, w)
    if conv_layer.bias is not None:
        out = out + conv_layer.bias.view(1, c_out, 1, 1)

    ad = alpha.detach()
    valid_sum = ad.squeeze(1).sum(dim=1)  # [B, H*W]
    n_val = in_bounds_mask.sum(dim=1)  # [1, H*W]
    center_idx = (h // 2) * w + (w // 2)
    center_tap = (k * k) // 2
    stats = {
        f"alpha_k{k}_valid_mean": float((valid_sum / n_val).mean()),
        f"alpha_k{k}_max": float(ad.max()),
        f"alpha_k{k}_center_self": float(ad[:, 0, center_tap, center_idx].mean()),
    }
    return out, stats


class SRA0C02FusionStage(nn.Module):
    def __init__(self, original, pool, group, seed, window_size=11):
        super().__init__()
        if group not in GROUPS:
            raise ValueError(f"Unknown SRA0C02 group: {group}")
        self.original = original
        self.group = group
        self.seed = int(seed)
        object.__setattr__(self, "_sr_pool", pool)  # Reference native SR pool without duplicate registration.
        self.disabled = False
        self.step = 0
        self.stats = {}
        self.history = []

        ones = torch.ones((1, 1, window_size, window_size), dtype=torch.float32)
        self.register_buffer("mask_k3", F.unfold(ones, kernel_size=3, padding=1), persistent=False)
        self.register_buffer("mask_k5", F.unfold(ones, kernel_size=5, padding=2), persistent=False)
        self.register_buffer("fusion_permutation", fixed_derangement(window_size * window_size), persistent=False)

        if group in ("R00", "R01", "R02"):
            # Minimal single-scale 1:1 direct(64) + 3x3(64) cross-modal neighborhood fusion on A0 (134,519 params).
            # R00, R01, R02 share bit-identical architecture, parameter count, and initial weights!
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(self.seed + 2026110810)
                self.direct_1x1 = nn.Sequential(
                    nn.Conv2d(128, 64, 1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1),
                )
                self.compress_1x1 = nn.Sequential(
                    nn.Conv2d(128, 64, 1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                )
                self.conv3x3 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
                self.post_3x3 = nn.Sequential(
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1),
                )
        else:
            # R03, R04, R05 compose CFTransfer("C02") under the exact same generator state as TransMixer_FV3_C02,
            # guaranteeing exact 680,311 parameters (+0 added) and bit-identical initial weights to historical C02!
            self.interaction = CFTransfer("C02")

    def record_gradient(self, key, grad):
        name, value = key + "_gradient_norm", float(grad.norm())
        self.stats[name] = value
        if self.history and self.history[-1]["step"] == self.step:
            self.history[-1][name] = value

    def _select_fusion_prior(self, w_sr):
        num_px = float(w_sr.shape[-2] * w_sr.shape[-1])
        if self.group in ("R00", "R03"):
            return torch.full_like(w_sr.detach(), 1.0 / num_px)
        if self.group in ("R01", "R04"):
            return w_sr.detach()
        if self.group in ("R02", "R05"):
            bsz, ch, height, width = w_sr.shape
            return w_sr.detach().flatten(2)[:, :, self.fusion_permutation].reshape(bsz, ch, height, width)
        raise RuntimeError(f"Unhandled group: {self.group}")

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

        w_fusion = self._select_fusion_prior(w_sr)
        w_ro = self._sr_pool.weights.detach().clamp_min(1e-12)
        ro_ent = float((-(w_ro * w_ro.log()).sum(dim=(2, 3))).mean())

        if self.group in ("R00", "R01", "R02"):
            x_in = torch.cat((h_feat, l_feat), dim=1)
            y0 = self.direct_1x1(x_in)
            z = self.compress_1x1(x_in)
            conv_out, st3 = sr_modulated_conv2d(z, self.conv3x3, self.mask_k3, w_fusion)
            y1 = self.post_3x3(conv_out)
            u = torch.cat((y0, y1), dim=1)
            self.stats.update(
                **st3,
                alpha_k5_valid_mean=1.0,
                alpha_k5_max=1.0,
                alpha_k5_center_self=1.0,
                readout_entropy=ro_ent,
            )
            if self.training:
                self.step += 1
            return self.original(u)

        # C02 family (R03, R04, R05): exact CrossModalRelations + 1x1 + SR-modulated 3x3 & 5x5 average
        h_rel, l_rel = self.interaction.relations(h_feat, l_feat)
        x_rel = torch.cat((h_rel, l_rel), dim=1)
        b1 = self.interaction.branches["1"](x_rel)

        seq3 = self.interaction.branches["3"]
        c3, st3 = sr_modulated_conv2d(x_rel, seq3[0], self.mask_k3, w_fusion)
        b3 = seq3[1:](c3)

        seq5 = self.interaction.branches["5"]
        c5, st5 = sr_modulated_conv2d(x_rel, seq5[0], self.mask_k5, w_fusion)
        b5 = seq5[1:](c5)

        mixed = (b1 + b3 + b5) / 3.0
        self.stats.update(**st3, **st5, readout_entropy=ro_ent)
        if self.training:
            self.step += 1
        return self.original(mixed)

    def record_loss(self, ce):
        self.stats.update(step=self.step, ce=float(ce.detach()), auxiliary_loss=0.0)
        self.history.append(dict(self.stats))
        return ce
