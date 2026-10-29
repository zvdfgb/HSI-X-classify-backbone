"""Native lab model adapter and optional loss/finish hooks."""
from pathlib import Path

import torch
from torch.nn import functional as F

from ..transmixer_net.model import TransMixerNet
from .model import NeighborTransMixer, ConceptTransMixer, FrozenCorrector, state_digest


VARIANTS = ("SR1", "SR2", "SR3", "PN1", "PN2", "OC1", "OC2", "OC3")
MODEL_NAMES = {key: f"TransMixer_Paper_{key}" for key in VARIANTS}


def build_model(config, dataset_type, device):
    manager = getattr(config, "_global_config_manager", None)
    exp = manager.config if manager is not None else getattr(config, "config", config)
    variant = next((key for key, name in MODEL_NAMES.items() if name == exp.model_name), None)
    if variant is None or exp.ablation_mode != "decoupled" or dataset_type != 0:
        raise ValueError("Paper experiments require fixed-1:1 Houston2013 and a registered variant")
    settings = dict(exp.paper_transfer or {})
    kwargs = dict(hsi_channels=config.get_value("channels"),
                  aux_channels=config.get_value("lidar_or_sar_channels")[dataset_type],
                  num_classes=config.get_value("out_features")[dataset_type],
                  hidden_dim=64, num_blocks=2, window_size=config.get_value("windowSize"))
    if variant.startswith("SR"):
        net = NeighborTransMixer(variant, spectral_scale=settings.get("spectral_scale", 1.),
                                 height_scale=settings.get("height_scale", 1.),
                                 spatial_sigma=settings.get("spatial_sigma", 3.), **kwargs)
    elif variant.startswith("OC"):
        net = ConceptTransMixer(variant, concept_mean=settings.get("concept_mean", [0., 0.]),
                                concept_std=settings.get("concept_std", [1., 1.]), **kwargs)
    else:
        from ...fusion_experiment import sha256
        path = Path(settings["baseline_checkpoint"])
        if sha256(path) != settings["baseline_sha256"]:
            raise ValueError("Frozen baseline checkpoint SHA256 mismatch")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("selection_split") != "validation":
            raise ValueError("Corrector baseline must have validation-selected provenance")
        base = TransMixerNet(**kwargs)
        base.load_state_dict(checkpoint["state_dict"], strict=True)
        net = FrozenCorrector(variant, base, settings.get("correction_hidden", 16))
    bundle = {"net": net.to(device), "variant": variant, "settings": settings}
    if variant.startswith("PN"):
        bundle["baseline_initial_sha256"] = state_digest(net.base)
    return bundle


def forward_train(bundle, batch):
    return bundle["net"](batch["hsi_pca"], batch["aux"])


def forward_eval(bundle, batch):
    return forward_train(bundle, batch)


def compute_loss(bundle, logits, batch, criterion):
    loss = criterion(logits, batch["label"])
    if bundle["variant"] in ("OC1", "OC3"):
        net = bundle["net"]
        targets = net.normalized_targets(batch["aux"]).detach()
        loss = loss + bundle["settings"].get("concept_weight", .1) * F.mse_loss(net.classifier.last_concepts, targets)
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite paper-transfer objective")
    return loss


def on_training_complete(bundle):
    if bundle["variant"].startswith("PN"):
        base = bundle["net"].base
        unchanged = state_digest(base) == bundle["baseline_initial_sha256"]
        frozen = all(not p.requires_grad and p.grad is None for p in base.parameters())
        if not unchanged or not frozen:
            raise RuntimeError("Frozen baseline parameters/statistics/gradient audit failed")
