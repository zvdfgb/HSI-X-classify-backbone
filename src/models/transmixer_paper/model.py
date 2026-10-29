"""Reuse the original forward path; changes are confined to pooling/readout.

No SSM, local/global block, stem, or fusion operator is copied or replaced.
Extra learned modules are initialized inside fork_rng for paired comparisons.
"""
import hashlib

import torch
from torch import nn

from ..transmixer_net.model import TransMixerNet


def state_digest(net):
    digest = hashlib.sha256()
    for name, value in net.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def geometry_targets(aux):
    """Two geometric proxies, computed only from the input LiDAR patch."""
    if aux.shape[1] != 1:
        raise ValueError("Height concepts require the one-channel Houston LiDAR input")
    flat = aux[:, 0].flatten(1)
    center = aux[:, 0, aux.shape[-2] // 2, aux.shape[-1] // 2]
    return torch.stack((center - flat.mean(1), flat.std(1, unbiased=False)), 1)


class WeightedPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = None

    def forward(self, features):
        if self.weights is None or self.weights.shape[-2:] != features.shape[-2:]:
            raise RuntimeError("Input-derived pooling weights were not prepared")
        return (features * self.weights).sum((2, 3), keepdim=True)


class NeighborTransMixer(TransMixerNet):
    def __init__(self, variant, spectral_scale=1., height_scale=1., spatial_sigma=3., **kwargs):
        super().__init__(**kwargs)
        if variant not in ("SR1", "SR2", "SR3"):
            raise ValueError(variant)
        self.variant = variant
        self.gap = WeightedPool()
        self.register_buffer("spectral_scale", torch.tensor(float(spectral_scale)))
        self.register_buffer("height_scale", torch.tensor(float(height_scale)))
        self.spatial_sigma = float(spatial_sigma)

    def neighbor_weights(self, hsi, aux):
        hsi = hsi[:, 0] if hsi.ndim == 5 else hsi
        height, width = hsi.shape[-2:]
        yy = torch.arange(height, device=hsi.device, dtype=hsi.dtype) - height // 2
        xx = torch.arange(width, device=hsi.device, dtype=hsi.dtype) - width // 2
        score = -(yy[:, None].square() + xx[None, :].square()) / (2 * self.spatial_sigma ** 2)
        score = score[None].expand(hsi.shape[0], -1, -1)
        if self.variant in ("SR2", "SR3"):
            diff = hsi - hsi[:, :, height//2:height//2+1, width//2:width//2+1]
            score = score - diff.square().mean(1) / self.spectral_scale.clamp_min(1e-8)
        if self.variant == "SR3":
            diff = aux - aux[:, :, height//2:height//2+1, width//2:width//2+1]
            score = score - diff.square().mean(1) / self.height_scale.clamp_min(1e-8)
        return score.flatten(1).softmax(1).reshape(-1, 1, height, width)

    def forward(self, hsi_pca, aux):
        with torch.no_grad():
            self.gap.weights = self.neighbor_weights(hsi_pca, aux)
        return super().forward(hsi_pca, aux)


class ConceptReadout(nn.Module):
    def __init__(self, original, dim, feed_concepts):
        super().__init__()
        self.concept_head = nn.Linear(dim, 2)
        self.feed_concepts = feed_concepts
        self.last_concepts = None
        if feed_concepts:
            old = original[0]
            expanded = nn.Linear(dim + 2, old.out_features)
            with torch.no_grad():
                expanded.weight[:, :dim].copy_(old.weight)
                expanded.bias.copy_(old.bias)
            original[0] = expanded
        self.classification = original
        self.replace_concepts_with_mean = False

    def forward(self, pooled):
        concepts = self.concept_head(pooled)
        self.last_concepts = concepts
        if self.feed_concepts:
            used = torch.zeros_like(concepts) if self.replace_concepts_with_mean else concepts
            pooled = torch.cat((pooled, used), 1)
        return self.classification(pooled)


class ConceptTransMixer(TransMixerNet):
    def __init__(self, variant, concept_mean=(0., 0.), concept_std=(1., 1.), **kwargs):
        super().__init__(**kwargs)
        if variant not in ("OC1", "OC2", "OC3"):
            raise ValueError(variant)
        self.variant = variant
        with torch.random.fork_rng(devices=[]):
            self.classifier = ConceptReadout(self.classifier, self.hidden_dim, variant != "OC1")
        self.register_buffer("concept_mean", torch.tensor(concept_mean, dtype=torch.float32))
        self.register_buffer("concept_std", torch.tensor(concept_std, dtype=torch.float32).clamp_min(1e-8))

    def normalized_targets(self, aux):
        return (geometry_targets(aux) - self.concept_mean) / self.concept_std


class FrozenCorrector(nn.Module):
    def __init__(self, variant, baseline, correction_hidden=16):
        super().__init__()
        if variant not in ("PN1", "PN2"):
            raise ValueError(variant)
        self.variant, self.base = variant, baseline
        self.base.requires_grad_(False)
        self.base.eval()
        dim = self.base.hidden_dim
        with torch.random.fork_rng(devices=[]):
            self.corrector = nn.Sequential(nn.Linear(3 * dim, correction_hidden), nn.ReLU(),
                                           nn.Linear(correction_hidden, dim))
            nn.init.zeros_(self.corrector[-1].weight)
            nn.init.zeros_(self.corrector[-1].bias)
        # Hooks read the existing backbone's intermediates; its forward is unchanged.
        self.base.fusion_conv.register_forward_pre_hook(self._capture_modalities)
        self.base.gap.register_forward_hook(self._capture_pooled)
        self._modalities = None
        self._pooled = None

    def _capture_modalities(self, module, args):
        h, a = args[0].chunk(2, 1)
        self._modalities = (h.mean((2, 3)), a.mean((2, 3)))

    def _capture_pooled(self, module, args, output):
        self._pooled = output.flatten(1)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()  # Includes frozen BN statistics and classifier dropout.
        return self

    def forward(self, hsi_pca, aux):
        self.base.eval()
        with torch.no_grad():
            self.base(hsi_pca, aux)
        z = self._pooled
        h, a = self._modalities
        inputs = (z, z, z) if self.variant == "PN1" else (h, a, z)
        delta = self.corrector(torch.cat(inputs, 1))
        return self.base.classifier(z + delta)
