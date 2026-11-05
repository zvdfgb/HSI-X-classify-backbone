"""Fusion-only transfers defined in the approved 20261104 review document.

Native encoders, fusion_conv, SR2, classifier and training protocol are composed,
not copied. This is explicitly NOT full reproduction of any source paper.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

GROUPS = tuple(f"{family}{i:02d}" for family in "CSI" for i in range(1, 5))


class CrossModalRelations(nn.Module):
    """Author CF-IPT matrix directions, with the two modalities mapped to 64D."""
    def __init__(self):
        super().__init__()
        self.value = nn.Conv2d(128, 128, 1)
        self.h_key = nn.Conv2d(64, 64, 1)
        self.l_key = nn.Conv2d(64, 64, 1)

    def forward(self, h, l):
        vh, vl = self.value(torch.cat((h, l), 1)).flatten(2).chunk(2, 1)
        spatial = (l.flatten(2).transpose(1, 2) @ self.h_key(h).flatten(2)).softmax(-1)
        channel = (h.flatten(2) @ self.l_key(l).flatten(2).transpose(1, 2)).softmax(-1)
        return h + (channel @ vh).reshape_as(h), l + (vl @ spatial).reshape_as(l)


class CFTransfer(nn.Module):
    def __init__(self, group):
        super().__init__()
        self.relations = CrossModalRelations()
        self.branches = nn.ModuleDict()
        # Component-local RNG streams pair shared weights across all four groups.
        for k in ((1, 3, 5) if group in ("C02", "C04") else (3,)):
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(torch.initial_seed() + 2026110400 + k)
                self.branches[str(k)] = nn.Sequential(nn.Conv2d(128, 128, k, padding=k//2),
                    nn.ReLU(), nn.BatchNorm2d(128), nn.Dropout(0.1))
        self.gate = (nn.Sequential(nn.Conv2d(128, 64, 1), nn.ReLU(), nn.Conv2d(64, 128, 1))
                     if group in ("C03", "C04") else None)
        self.stats = {}

    def forward(self, h, l):
        h, l = self.relations(h, l)
        x = torch.cat((h, l), 1)
        g = (128 * (self.gate(F.adaptive_avg_pool2d(x, 1)) +
                    self.gate(F.adaptive_max_pool2d(x, 1))).softmax(1)
             if self.gate is not None else torch.ones_like(x[:, :, :1, :1]))
        self.stats = {"channel_gate_min": float(g.detach().min()),
                      "channel_gate_max": float(g.detach().max()),
                      "channel_gate_mean": float(g.detach().mean())}
        return sum(branch(x*g) for branch in self.branches.values()) / len(self.branches)


class CrossAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q, self.k, self.v, self.out = (nn.Conv2d(64, 64, 1) for _ in range(4))

    def forward(self, x, other):
        b, _, height, width = x.shape
        context = F.adaptive_avg_pool2d(other, (4, 4))
        q = self.q(x).reshape(b, 4, 16, -1).transpose(-1, -2)
        k = self.k(context).reshape(b, 4, 16, -1)
        v = self.v(context).reshape(b, 4, 16, -1).transpose(-1, -2)
        y = ((q @ k / 4).softmax(-1) @ v).transpose(-1, -2).reshape(b, 64, height, width)
        return self.out(y)


class CFI(nn.Module):
    def __init__(self, fixed_gate=False):
        super().__init__()
        self.h_to_l, self.l_to_h = CrossAttention(), CrossAttention()
        self.gate = None if fixed_gate else nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, h, l):
        c = self.l_to_h(h, l) + self.h_to_l(l, h)
        g = torch.full_like(c[:, :1], .5) if self.gate is None else self.gate(c).sigmoid()
        y = g*c + (1-g)*(h+l)
        residual = (y-h-l)/2
        return torch.cat((h+residual, l+residual), 1), y, g


def intervention_masks(batch, seed, step, device):
    """Independent stream: same seed/epoch/batch => identical S03/S04 masks."""
    gen = torch.Generator().manual_seed(20261104 + seed*10000000 + step)
    positions = [(y, x) for y in range(9) for x in range(9)
                 if not (y <= 5 < y+3 and x <= 5 < x+3)]
    paired = [(i, j) for i, (y, x) in enumerate(positions)
              for j, (v, u) in enumerate(positions)
              if y+3 <= v or v+3 <= y or x+3 <= u or u+3 <= x]
    pair_ids = torch.randint(len(paired), (batch,), generator=gen).tolist()
    common_ids = torch.randint(len(positions), (batch,), generator=gen).tolist()
    masks = [torch.ones(batch, 1, 11, 11) for _ in range(3)]
    for b, (pair_id, common_id) in enumerate(zip(pair_ids, common_ids)):
        i, j = paired[pair_id]
        for mask, index in zip(masks, (i, j, common_id)):
            y, x = positions[index]
            mask[b, :, y:y+3, x:x+3] = 0
    return tuple(m.to(device) for m in masks)


def gaussian_projection():
    gen = torch.Generator().manual_seed(20261104)
    return torch.randn(32, 64, generator=gen)/math.sqrt(32)


def sparse_frequency(x):
    spec = torch.fft.rfft(x, dim=-1, norm="ortho")
    chosen = spec[:, 1:32].abs().square().topk(15, dim=-1).indices + 1
    mask = torch.zeros_like(spec.real).scatter(1, chosen, 1)
    mask[:, 0] = mask[:, 32] = 1
    return torch.fft.irfft(spec*mask, n=64, dim=-1, norm="ortho")


def generalized_kl(x, y):
    return (x*(x.clamp_min(1e-30).log()-y.clamp_min(1e-30).log())-x+y).sum()


def unbalanced_ot(h, l):
    """50 differentiable log-domain Sinkhorn iterations, fixed eps/rho."""
    cost = torch.cdist(h, l, p=2, compute_mode="donot_use_mm_for_euclid_dist")
    log_a = cost.new_full((len(h),), -math.log(len(h)))
    log_b = cost.new_full((len(l),), -math.log(len(l)))
    log_ref = log_a[:, None]+log_b[None, :]
    log_k = log_ref-cost/.05
    log_u, log_v = torch.zeros_like(log_a), torch.zeros_like(log_b)
    for _ in range(50):
        log_u = (log_a-torch.logsumexp(log_k+log_v[None, :], 1))/1.05
        log_v = (log_b-torch.logsumexp(log_k+log_u[:, None], 0))/1.05
    t = (log_k+log_u[:, None]+log_v[None, :]).exp()
    objective = (t*cost).sum()+.05*generalized_kl(t, log_ref.exp())
    objective = objective+generalized_kl(t.sum(1), log_a.exp())+generalized_kl(t.sum(0), log_b.exp())
    return objective, {"ot_cost_mean": float(cost.detach().mean()),
                       "ot_cost_min": float(cost.detach().min()), "ot_cost_max": float(cost.detach().max()),
                       "transport_mass": float(t.detach().sum())}


class FusionStage(nn.Module):
    def __init__(self, original, pool, group, seed):
        super().__init__()
        self.original, self.group, self.seed = original, group, seed
        object.__setattr__(self, "_sr_pool", pool)  # Same original pool, no duplicate registration.
        self.disabled, self.step, self.cache = False, 0, None
        self.stats, self.history = {}, []
        if group.startswith("C"):
            self.interaction = CFTransfer(group)
        elif group.startswith("S"):
            self.interaction = CFI(group == "S01")
        else:
            self.interaction = nn.ModuleList([nn.Conv2d(64, 64, 1) for _ in range(2)])
            for layer in self.interaction:
                nn.init.eye_(layer.weight[:, :, 0, 0]); nn.init.zeros_(layer.bias)
            if group == "I04":
                self.register_buffer("random_projection", gaussian_projection())

    def forward(self, x):
        self.cache = None
        if self.disabled:
            return self.original(x)
        h, l = x.chunk(2, 1)
        if self.training:
            for name, z in (("h", h), ("l", l)):
                if z.requires_grad:
                    z.register_hook(lambda grad, key=name: self.record_gradient(key, grad))
        if self.group.startswith("C"):
            mixed = self.interaction(h, l)
            self.stats.update(self.interaction.stats)
        elif self.group.startswith("S"):
            mixed, y, g = self.interaction(h, l)
            self.stats.update(gate_mean=float(g.detach().mean()),
                              gate_entropy=float(self.entropy(g.detach()).mean()))
            if self.training and self.group in ("S03", "S04"):
                a, b, same = intervention_masks(len(h), self.seed, self.step, h.device)
                views = [self.interaction(h*a, l*b), self.interaction(h*same, l*same)]
                self.cache = (y, [(v[1], v[2]) for v in views])
        else:
            h, l = [proj(z) for proj, z in zip(self.interaction, (h, l))]
            mixed = torch.cat((h, l), 1)
            if self.training and self.group != "I01":
                weights = self._sr_pool.weights
                if weights is None:
                    raise RuntimeError("Native SR2 weights have not been prepared")
                self.cache = tuple(F.normalize((z*weights).sum((2, 3)), dim=1) for z in (h, l))
        if self.training:
            self.step += 1
        return self.original(mixed)

    @staticmethod
    def entropy(g):
        g = g.clamp(1e-6, 1-1e-6)
        return -g*g.log()-(1-g)*(1-g).log()

    def record_gradient(self, key, grad):
        name, value = key+"_gradient_norm", float(grad.norm())
        self.stats[name] = value
        if self.history and self.history[-1]["step"] == self.step:
            self.history[-1][name] = value

    def additional_loss(self, ce):
        extra = ce.new_zeros(())
        cache, self.cache = self.cache, None  # No autograd graphs serialized with whole-model checkpoints.
        if self.group in ("S03", "S04"):
            y, views = cache
            reference = y.detach()
            rms = reference.square().mean(1, keepdim=True).sqrt()+1e-6
            terms = []
            for corrupted, g in views:
                difference = (corrupted-reference).abs().mean(1, keepdim=True)/rms
                if self.group == "S03":
                    terms.append(difference.mean())
                else:
                    area = (g.mean((1, 2, 3))-.3).abs().mean()
                    terms.append((g*difference).mean()+.1*area-.05*self.entropy(g).mean())
            extra = .1*torch.stack(terms).mean()
        elif self.group in ("I02", "I03", "I04"):
            h, l = cache
            if self.group == "I02":
                hc, lc = h-h.mean(0), l-l.mean(0)
                cov_h, cov_l = hc.T@hc/(len(h)-1), lc.T@lc/(len(l)-1)
                extra = .01*((h.mean(0)-l.mean(0)).square().mean()+(cov_h-cov_l).square().mean())
            else:
                if self.group == "I04":
                    self.stats.update(full64_pair_cost_mean=float(torch.cdist(h.detach(), l.detach()).mean()))
                    h, l = [sparse_frequency(z)@self.random_projection.T for z in (h, l)]
                objective, stats = unbalanced_ot(h, l)
                self.stats.update(stats)
                extra = .01*objective
        self.stats.update(step=self.step, ce=float(ce.detach()), auxiliary_loss=float(extra.detach()))
        self.history.append(dict(self.stats))
        return ce+extra
