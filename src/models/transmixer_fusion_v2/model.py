"""Fusion-only adaptations; native stems, backbone, SR and head are composed.

Scientific scope and differences from the papers are recorded in the approved
20261102 experiment design. No loader, optimizer, training or metric lives here.
FFT convention follows the lab's dffnet.model (spatial rFFT, ortho, explicit s).
UMF mechanisms: lfvision/Unaligned-RGB-T-Salient-Object-Detection.
FTRE objectives: Soceremite/FINE, adapted to two modalities and single AdamW.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call


GROUPS = tuple(f"{family}{i:02d}" for family in "FUD" for i in range(1, 5))


def conv_mlp(nin, nout, hidden=64):
    return nn.Sequential(nn.Conv2d(nin, hidden, 1), nn.ReLU(), nn.Conv2d(hidden, nout, 1))


def frequency_masks(height, width, device):
    fy = torch.fft.fftfreq(height, device=device)
    fx = torch.fft.rfftfreq(width, device=device)
    radius = (fy[:, None].square() + fx[None, :].square()).sqrt()
    radius = radius / radius.max()
    masks = torch.stack((radius <= 1 / 3, (radius > 1 / 3) & (radius <= 2 / 3), radius > 2 / 3))
    if not masks.flatten(1).any(1).all():
        raise ValueError("Empty frequency band")
    return masks


def band_features(x):
    spec = torch.fft.rfft2(x, norm="ortho")
    masks = frequency_masks(*x.shape[-2:], x.device)
    return [torch.fft.irfft2(spec * mask, s=x.shape[-2:], norm="ortho") for mask in masks]


class FrequencyInteraction(nn.Module):
    def __init__(self, group, dim=64):
        super().__init__()
        self.group = group
        self.weights = conv_mlp(2 * dim, 2 * 3 * dim)
        self.projections = nn.ModuleList([nn.Sequential(nn.Conv2d(dim, dim, 1), nn.ReLU()) for _ in range(6)])
        self.outputs = nn.ModuleList([nn.Conv2d(dim, dim, 1) for _ in range(2)])
        self.stats = {}

    def forward(self, h, l):
        spectrum = torch.cat([torch.fft.rfft2(x, norm="ortho").abs().mean((-2, -1), keepdim=True)
                              for x in (h, l)], 1)
        w = self.weights(spectrum).reshape(h.shape[0], 2, 3, h.shape[1], 1, 1).softmax(1)
        hb, lb = ([h] * 3, [l] * 3) if self.group == "F01" else (band_features(h), band_features(l))
        mixed = sum(w[:, 0, i] * self.projections[i](hb[i]) +
                    w[:, 1, i] * self.projections[3 + i](lb[i]) for i in range(3))
        self.stats = {"band_modality_weights": w.detach().mean((0, 3, 4, 5)).cpu().tolist(),
                      "h_band_energy": [float(x.detach().square().mean()) for x in hb],
                      "l_band_energy": [float(x.detach().square().mean()) for x in lb]}
        return self.outputs[0](mixed), self.outputs[1](mixed)


class DetailCompensation(nn.Module):
    def __init__(self, high_frequency, dim=64):
        super().__init__()
        self.high_frequency = high_frequency
        self.projections = nn.ModuleList([nn.Conv2d(dim, dim, 1) for _ in range(2)])
        self.weights = conv_mlp(dim, 2 * dim)

    def forward(self, coarse, shallow):
        source = [band_features(x)[2] for x in shallow] if self.high_frequency else shallow
        spectrum = torch.fft.rfft2(coarse, norm="ortho").abs().mean((-2, -1), keepdim=True)
        w = self.weights(spectrum).reshape(coarse.shape[0], 2, coarse.shape[1], 1, 1).softmax(1)
        return sum(w[:, i] * self.projections[i](source[i]) for i in range(2))


class GaussianHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1, groups=64), nn.ReLU())
        self.mu, self.lv = nn.Conv2d(64, 32, 1), nn.Conv2d(64, 32, 1)

    def forward(self, x):
        x = self.pre(x)
        return self.mu(x), self.lv(x).clamp(-10, 10)


class ReliabilityInteraction(nn.Module):
    def __init__(self, group):
        super().__init__()
        self.group = group
        self.heads = nn.ModuleList([GaussianHead(), GaussianHead()])
        self.conditional = conv_mlp(64, 64, 32)
        self.guide = nn.Conv2d(32, 64, 1)
        self.film = conv_mlp(64, 128)
        self.prior = nn.Sequential(nn.Conv2d(64, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 1, 1), nn.Sigmoid())
        self.confidence = (None if group == "U02" else
                           nn.Sequential(nn.Conv2d(2, 8, 1), nn.ReLU(), nn.Conv2d(8, 1, 1), nn.Sigmoid()))
        self.kl = None
        self.stats = {}

    def forward(self, h, l):
        pairs = [head(x) for head, x in zip(self.heads, (h, l))]
        if self.group == "U01":
            # Both heads contribute to the deterministic representation and gate.
            z = [mu + lv.tanh() for mu, lv in pairs]
            fused_mu, fused_lv = self.conditional(torch.cat(z, 1)).chunk(2, 1)
            guide = self.guide(fused_mu + fused_lv.tanh())
            confidence = self.confidence(torch.cat([q.tanh().mean(1, keepdim=True) for q in z], 1))
            self.kl = None
        else:
            z = [mu + (0.5 * lv).exp() * torch.randn_like(mu) if self.training else mu for mu, lv in pairs]
            fused_mu, fused_lv = self.conditional(torch.cat(z, 1)).chunk(2, 1)
            fused_lv = fused_lv.clamp(-10, 10)
            latent = (fused_mu + (0.5 * fused_lv).exp() * torch.randn_like(fused_mu)
                      if self.training else fused_mu)
            guide = self.guide(latent)
            self.kl = (torch.stack([0.5 * (lv.float().exp() + mu.float().square() - 1 - lv.float()).mean()
                                    for mu, lv in pairs]).mean() if self.training else None)
            if self.confidence is None:
                confidence = torch.ones_like(h[:, :1])
            else:
                # Official uncertainty head: exp(softplus(logvar)), inverse channel mean.
                inv_variance = [F.softplus(lv).exp().mean(1, keepdim=True).reciprocal() for _, lv in pairs]
                confidence = self.confidence(torch.cat(inv_variance, 1))
                if self.group == "U04":
                    confidence = confidence.mean((-2, -1), keepdim=True).expand_as(h[:, :1])
        gamma, beta = self.film(guide.mean((-2, -1), keepdim=True)).chunk(2, 1)
        # CGM: confidence controls FiLM and spatial prior; returns a residual update.
        update = self.prior(guide) * confidence * ((h * gamma + beta) * confidence - h)
        self.stats = {"confidence_mean": float(confidence.detach().mean()),
                      "confidence_spatial_std": float(confidence.detach().std((-2, -1), unbiased=False).mean()),
                      "logvar_mean": float(torch.stack([lv.detach().mean() for _, lv in pairs]).mean()),
                      "kl": float(self.kl.detach()) if self.kl is not None else 0.0}
        return update


class PairCritic(nn.Module):
    """NCE-CLUB pair scores with explicitly isolated representation gradients."""
    def __init__(self, dim=32):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(2 * dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def scores(self, x, y, frozen=False):
        n = x.shape[0]
        xy = torch.cat((x[:, None].expand(n, n, -1), y[None].expand(n, n, -1)), -1)
        if frozen:
            params = {k: p.detach() for k, p in self.score.named_parameters()}
            return functional_call(self.score, params, (xy,)).squeeze(-1) / 0.1
        return self.score(xy).squeeze(-1) / 0.1

    def objectives(self, x, y, upper=False):
        frozen = self.scores(x, y, frozen=True)
        # InfoNCE negative lower bound, or NCE-CLUB upper bound to minimize.
        representation = (frozen.diagonal().mean() - frozen.mean() if upper else
                          F.cross_entropy(frozen, torch.arange(x.shape[0], device=x.device)) - math.log(x.shape[0]))
        fitted = self.scores(x.detach(), y.detach())
        fit = F.cross_entropy(fitted, torch.arange(x.shape[0], device=x.device)) - math.log(x.shape[0])
        return representation, fit


class FactorizedInteraction(nn.Module):
    def __init__(self, group):
        super().__init__()
        self.group = group
        self.projections = nn.ModuleList([conv_mlp(64, 32) for _ in range(4)])
        self.refiners = nn.ModuleList([conv_mlp(32, 32, 32) for _ in range(4)])
        self.outputs = nn.ModuleList([nn.Conv2d(64, 64, 1) for _ in range(2)])
        if group != "D01":
            self.reconstruct = nn.ModuleList([conv_mlp(64, 64) for _ in range(2)])
            self.aux_heads = nn.ModuleList([nn.Linear(32, 15) for _ in range(4)])
        if group in ("D03", "D04"):
            self.shared_critic, self.unique_critic = PairCritic(), PairCritic()
        if group == "D04":
            self.label_embedding = nn.Embedding(15, 32)
            self.conditional_critic = PairCritic()
            self.task_critics = nn.ModuleList([PairCritic(), PairCritic()])
        self.cache = None
        self.stats = {}

    def forward(self, h, l):
        raw = [p(x) for p, x in zip(self.projections, (h, h, l, l))]
        refined = [p(x) for p, x in zip(self.refiners, raw)]
        if self.training:
            self.cache = (h, l, raw, refined)
        else:
            self.cache = None
        return self.outputs[0](torch.cat(refined[:2], 1)), self.outputs[1](torch.cat(refined[2:], 1))

    def loss(self, criterion, labels, weights, zero):
        if self.cache is None:
            raise RuntimeError("Training-only factorization cache missing")
        h, l, raw, refined = self.cache
        self.cache = None  # No autograd graph or training labels in serialized net.
        if self.group == "D01":
            self.stats = {}
            return zero
        rec = torch.stack([F.mse_loss(head(torch.cat(parts, 1)), target.detach()) for head, parts, target in
                           zip(self.reconstruct, (raw[:2], raw[2:]), (h, l))]).mean()
        pooled = [(x * weights).sum((-2, -1)) for x in refined]
        aux = torch.stack([criterion(head(x), labels) for head, x in zip(self.aux_heads, pooled)]).mean()
        total = 0.1 * aux + 0.01 * rec
        terms = {"aux_ce": float(aux.detach()), "reconstruction": float(rec.detach())}
        if self.group in ("D03", "D04"):
            raw_pooled = [(x * weights).sum((-2, -1)) for x in raw]
            shared, shared_fit = self.shared_critic.objectives(raw_pooled[0], raw_pooled[2])
            unique, unique_fit = self.unique_critic.objectives(raw_pooled[1], raw_pooled[3], upper=True)
            dec, fit = (shared + unique) / 2, (shared_fit + unique_fit) / 2
            total = total + 0.01 * (dec + fit)
            terms.update(decomposition_mi=float(dec.detach()), decomposition_critic_fit=float(fit.detach()))
        if self.group == "D04":
            conditional, conditional_fit, valid = [], [], 0
            for label in labels.unique():
                indices = (labels == label).nonzero().flatten()
                if indices.numel() >= 2:
                    a, b = self.conditional_critic.objectives(pooled[0][indices], pooled[2][indices], upper=True)
                    conditional.append(a * indices.numel())
                    conditional_fit.append(b * indices.numel())
                    valid += indices.numel()
            c = sum(conditional) / valid if valid else zero
            cf = sum(conditional_fit) / valid if valid else zero
            label_repr = self.label_embedding(labels)
            task = [critic.objectives(x, label_repr) for critic, x in zip(self.task_critics, (pooled[1], pooled[3]))]
            task_mi = (c + task[0][0] + task[1][0]) / 3
            task_fit = (cf + task[0][1] + task[1][1]) / 3
            total = total + 0.01 * (task_mi + task_fit)
            terms.update(task_mi=float(task_mi.detach()), task_critic_fit=float(task_fit.detach()), conditional_valid_samples=valid)
        self.stats = terms
        return total


class FusionStage(nn.Module):
    """Wrap original fusion_conv and capture existing shallow outputs by hooks."""
    def __init__(self, original, group):
        super().__init__()
        self.original, self.group = original, group
        self.alpha_h = nn.Parameter(torch.zeros(()))
        if not group.startswith("U"):
            self.alpha_l = nn.Parameter(torch.zeros(()))
        self.interaction = (FrequencyInteraction(group) if group.startswith("F") else
                            ReliabilityInteraction(group) if group.startswith("U") else FactorizedInteraction(group))
        if group in ("F01", "F03", "F04"):
            self.compensation = DetailCompensation(group == "F04")
            self.alpha_detail = nn.Parameter(torch.zeros(()))
        self.shallow_h, self.shallow_l = None, None
        self.last_stats = {}

    def capture_h(self, module, args, output):
        self.shallow_h = output

    def capture_l(self, module, args, output):
        self.shallow_l = output

    def forward(self, joined):
        h, l = joined.chunk(2, 1)
        if self.group.startswith("U"):
            mh = self.interaction(h, l)
            ml = None
            updated = (h + self.alpha_h * mh, l)
        else:
            mh, ml = self.interaction(h, l)
            updated = (h + self.alpha_h * mh, l + self.alpha_l * ml)
        coarse = self.original(torch.cat(updated, 1))
        self.last_stats = {"h_update_rms": float((self.alpha_h * mh).detach().square().mean().sqrt())}
        if ml is not None:
            self.last_stats["l_update_rms"] = float((self.alpha_l * ml).detach().square().mean().sqrt())
        if hasattr(self, "compensation"):
            if self.shallow_h is None or self.shallow_l is None:
                raise RuntimeError("Shallow native stem hooks were not called")
            detail = self.alpha_detail * self.compensation(coarse, (self.shallow_h, self.shallow_l))
            self.last_stats["detail_rms"] = float(detail.detach().square().mean().sqrt())
            coarse = coarse + detail
        self.shallow_h = self.shallow_l = None
        return coarse

    def additional_loss(self, base_loss, criterion, labels, weights):
        if self.group.startswith("U"):
            kl = self.interaction.kl
            self.interaction.kl = None
            return base_loss if kl is None else base_loss + 0.001 * kl
        if self.group.startswith("D"):
            return base_loss + self.interaction.loss(criterion, labels, weights, base_loss.new_zeros(()))
        return base_loss

    def diagnostics(self):
        values = {"group": self.group, "alpha_h": float(self.alpha_h.detach()),
                  **self.last_stats, **self.interaction.stats}
        for key in ("alpha_l", "alpha_detail"):
            if hasattr(self, key):
                values[key] = float(getattr(self, key).detach())
        return values
