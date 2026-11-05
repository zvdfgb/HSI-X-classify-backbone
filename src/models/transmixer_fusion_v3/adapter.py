"""Native adapter: approved C01-C04/S01-S04/I01-I04, seeds6/7/8 only."""
import json
from pathlib import Path
import torch
from ..transmixer_paper.model import NeighborTransMixer
from .model import FusionStage, GROUPS


def build_model(config, dataset_type, device):
    manager = getattr(config, "_global_config_manager", None)
    exp = manager.config if manager is not None else getattr(config, "config", config)
    group = exp.model_name.removeprefix("TransMixer_FV3_")
    if group not in GROUPS or exp.model_name != f"TransMixer_FV3_{group}" or dataset_type != 0:
        raise ValueError("Only the approved Houston2013 FV3 matrix is supported")
    fixed = dict(channels=30, window_size=11, epochs=100, batch_size=128,
                 num_workers=0, learning_rate=.0001, ablation_mode="decoupled")
    if exp.random_seed not in (6, 7, 8) or any(getattr(exp, k) != v for k, v in fixed.items()):
        raise ValueError("Unapproved protocol or seed")
    if dict(exp.paper_transfer or {}) != {"group": group} or exp.distillation:
        raise ValueError("Unapproved fusion settings")
    if dict(exp.sr or {}) != {"spatial_sigma": 3., "spectral_scale": .2775005102157593}:
        raise ValueError("Original SR2 settings required")
    net = NeighborTransMixer("SR2", spectral_scale=exp.sr["spectral_scale"], spatial_sigma=3.,
        hsi_channels=30, aux_channels=1, num_classes=15, hidden_dim=64,
        num_blocks=2, window_size=11, mode="decoupled")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(exp.random_seed)
        net.fusion_conv = FusionStage(net.fusion_conv, net.gap, group, exp.random_seed)
    return {"net": net.to(device), "diagnostics_path": str(Path(exp.report_path).with_suffix(".diagnostics.json"))}


def forward_train(bundle, batch):
    return bundle["net"](batch["hsi_pca"], batch["aux"])


def forward_eval(bundle, batch):
    return forward_train(bundle, batch)


def compute_loss(bundle, outputs, batch, criterion):
    return bundle["net"].fusion_conv.additional_loss(criterion(outputs, batch["label"]))


def on_training_complete(bundle):
    stage = bundle["net"].fusion_conv
    path = Path(bundle["diagnostics_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"group": stage.group, "seed": stage.seed,
        "training_steps": stage.step, "history": stage.history}, ensure_ascii=False, indent=2), encoding="utf-8")
