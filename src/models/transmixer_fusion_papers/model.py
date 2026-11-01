"""Fusion-stage additions composed with the existing lab backbone.

No backbone forward, SSM recurrence, pooling, data or training loop is copied.
ConditionedProjection supplies external parameters to the ORIGINAL SSM forward.
"""
import torch
from torch import nn

from ..transmixer_net.model import SelectiveSSM2D


GROUPS = {
    "G02": ("scan", "independent", True),
    "G03": ("scan", "block_concat", True),
    "G04": ("scan", "interleave", True),
    "G05": ("scan", "interleave", False),
    "G06": ("condition", "self", True),
    "G07": ("condition", "cross_bc", True),
    "G08": ("condition", "cross_bc_delta", True),
    "G09": ("condition", "cross_bc_delta", False),
    "G10": ("experts", "uniform", True),
    "G11": ("experts", "dynamic", True),
    "G12": ("experts", "balanced", True),
    "G13": ("experts", "consensus", True),
    "G14": ("experts", "consensus", False),
}


class ConditionedProjection(nn.Module):
    """Wrap an existing x_proj; no new parameters or numerical recurrence."""

    def __init__(self, projection, dim, mode):
        super().__init__()
        self.projection = projection
        self.dim = dim
        self.mode = mode
        self.control = None

    def forward(self, target):
        if self.mode == "self":
            return self.projection(target)
        if self.control is None or self.control.shape != target.shape:
            raise ValueError("Conditional SSM requires aligned control tokens")
        external = self.projection(self.control)
        if self.mode == "cross_bc_delta":
            return external
        own_delta = self.projection(target)[..., :self.dim]
        return torch.cat((own_delta, external[..., self.dim:]), -1)


class ConditionedSSM(SelectiveSSM2D):
    """Reuse SelectiveSSM2D.forward byte-for-byte through a projection wrapper."""

    def __init__(self, dim, mode, state_dim=16):
        super().__init__(dim, d_state=state_dim)
        self.x_proj = ConditionedProjection(self.x_proj, dim, mode)

    def forward(self, target, control):
        self.x_proj.control = control
        try:
            return super().forward(target)
        finally:
            # Do not retain inputs in serialized whole-model checkpoints.
            self.x_proj.control = None


class ScanInteraction(nn.Module):
    def __init__(self, dim, order):
        super().__init__()
        self.order = order
        self.norm_h = nn.LayerNorm(dim)
        self.norm_l = nn.LayerNorm(dim)
        self.ssm = SelectiveSSM2D(dim, d_state=16)

    def bidirectional(self, sequence):
        fwd = self.ssm(sequence)[0]
        rev = self.ssm(sequence.flip(1))[0].flip(1)
        return (fwd + rev) * 0.5

    def forward(self, h, l):
        height, width = h.shape[-2:]
        ht = self.norm_h(h.flatten(2).transpose(1, 2))
        lt = self.norm_l(l.flatten(2).transpose(1, 2))
        if self.order == "independent":
            mh, ml = self.bidirectional(ht), self.bidirectional(lt)
        elif self.order == "block_concat":
            mh, ml = self.bidirectional(torch.cat((ht, lt), 1)).chunk(2, 1)
        else:
            sequence = torch.stack((ht, lt), 2).flatten(1, 2)
            result = self.bidirectional(sequence).reshape(ht.shape[0], ht.shape[1], 2, ht.shape[2])
            mh, ml = result[:, :, 0], result[:, :, 1]
        return (mh.transpose(1, 2).reshape_as(h), ml.transpose(1, 2).reshape_as(l))


class ConditionalInteraction(nn.Module):
    def __init__(self, dim, mode):
        super().__init__()
        self.norm_h = nn.LayerNorm(dim)
        self.norm_l = nn.LayerNorm(dim)
        self.h_ssm = ConditionedSSM(dim, mode)
        self.l_ssm = ConditionedSSM(dim, mode)

    @staticmethod
    def bidirectional(ssm, target, control):
        fwd = ssm(target, control)[0]
        rev = ssm(target.flip(1), control.flip(1))[0].flip(1)
        return (fwd + rev) * 0.5

    def forward(self, h, l):
        ht = self.norm_h(h.flatten(2).transpose(1, 2))
        lt = self.norm_l(l.flatten(2).transpose(1, 2))
        mh = self.bidirectional(self.h_ssm, ht, lt)
        ml = self.bidirectional(self.l_ssm, lt, ht)
        return mh.transpose(1, 2).reshape_as(h), ml.transpose(1, 2).reshape_as(l)


class ExpertInteraction(nn.Module):
    def __init__(self, dim, mode):
        super().__init__()
        self.mode = mode
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Conv2d(2 * dim, 16, 1), nn.GELU(), nn.Conv2d(16, 2 * dim, 1))
            for _ in range(4)
        ])
        self.router = nn.Sequential(nn.Linear(2 * dim, 16), nn.GELU(), nn.Linear(16, 4))
        if mode == "uniform":
            self.router.requires_grad_(False)
        self.balance = None
        self.consensus = None
        self.eval_stats = None

    def train(self, mode=True):
        if not mode:
            self.eval_stats = None
        return super().train(mode)

    def forward(self, h, l):
        joined = torch.cat((h, l), 1)
        summary = joined.mean((2, 3)) + joined.amax((2, 3))
        raw_weights = self.router(summary).softmax(1)
        weights = torch.full_like(raw_weights, 0.25) if self.mode == "uniform" else raw_weights
        outputs = torch.stack([expert(joined) for expert in self.experts], 1)
        mixture = (outputs * weights[:, :, None, None, None]).sum(1)
        units = outputs / (outputs.square().mean(2, keepdim=True) + 1e-6).sqrt()
        center = (units * weights[:, :, None, None, None]).sum(1, keepdim=True)
        # Train-only losses use the existing adapter.compute_loss extension.
        if self.training:
            self.balance = 4.0 * (weights.mean(0) - 0.25).square().sum()
            self.consensus = ((units - center).square().mean(2) * weights[:, :, None, None]).sum(1).mean()
        else:
            self.balance = self.consensus = None
            with torch.no_grad():
                vectors = torch.nn.functional.normalize(outputs.flatten(2), dim=2)
                similarities = torch.matmul(vectors, vectors.transpose(1, 2))
                mask = ~torch.eye(4, dtype=torch.bool, device=h.device)
                stats = {
                    "weight_sum": weights.sum(0),
                    "entropy_sum": -(weights * weights.clamp_min(1e-8).log()).sum(),
                    "rms_sum": outputs.square().mean((2, 3, 4)).sqrt().sum(0),
                    "similarity_sum": similarities[:, mask].mean(1).sum(),
                    "samples": h.new_tensor(h.shape[0]),
                }
                if self.eval_stats is None:
                    self.eval_stats = {k: v.detach() for k, v in stats.items()}
                else:
                    for key, value in stats.items():
                        self.eval_stats[key] += value.detach()
        return mixture.chunk(2, 1)

    def diagnostics(self):
        if self.eval_stats is None:
            return {}
        s = self.eval_stats
        count = float(s["samples"].cpu())
        return {
            "eval_samples": int(count),
            "mean_weights": (s["weight_sum"] / count).cpu().tolist(),
            "mean_weight_entropy": float((s["entropy_sum"] / count).cpu()),
            "mean_expert_rms": (s["rms_sum"] / count).cpu().tolist(),
            "mean_pairwise_cosine": float((s["similarity_sum"] / count).cpu()),
        }


class FusionStage(nn.Module):
    """Wrap only the original fusion_conv, preserving the original net forward."""

    def __init__(self, original, group, dim=64):
        super().__init__()
        self.original = original
        self.group = group
        family, variant, _ = GROUPS[group]
        self.family = family
        self.alpha_h = nn.Parameter(torch.zeros(()))
        self.alpha_l = nn.Parameter(torch.zeros(()))
        cls = {"scan": ScanInteraction, "condition": ConditionalInteraction, "experts": ExpertInteraction}[family]
        self.interaction = cls(dim, variant)
        self.lambda_balance = 0.01 if group in ("G12", "G13", "G14") else 0.0
        self.lambda_consensus = 0.001 if group in ("G13", "G14") else 0.0

    def forward(self, joined):
        h, l = joined.chunk(2, 1)
        mh, ml = self.interaction(h, l)
        return self.original(torch.cat((h + self.alpha_h * mh, l + self.alpha_l * ml), 1))

    def additional_loss(self, classification_loss):
        if self.family != "experts":
            return classification_loss
        loss = classification_loss
        if self.lambda_balance:
            loss = loss + self.lambda_balance * self.interaction.balance
        if self.lambda_consensus:
            loss = loss + self.lambda_consensus * self.interaction.consensus
        return loss

    def diagnostics(self):
        values = {"group": self.group, "alpha_h": float(self.alpha_h.detach().cpu()),
                  "alpha_l": float(self.alpha_l.detach().cpu())}
        if self.family == "experts":
            values.update(self.interaction.diagnostics())
        return values
