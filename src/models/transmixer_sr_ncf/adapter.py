"""Native adapter for approved SR-NCF matrix K00-K06 on Houston2013 (seeds 6, 7, 8)."""
import json
from pathlib import Path
import torch
from ..transmixer_paper.model import NeighborTransMixer
from .model import SRNCFFusionStage, GROUPS


def build_model(config, dataset_type, device):
    manager = getattr(config, "_global_config_manager", None)
    exp = manager.config if manager is not None else getattr(config, "config", config)
    group = exp.model_name.removeprefix("TransMixer_SRNCF_")
    if group not in GROUPS or exp.model_name != f"TransMixer_SRNCF_{group}" or dataset_type != 0:
        raise ValueError("Only the approved Houston2013 SR-NCF K00-K06 matrix is supported")
    fixed = dict(
        channels=30,
        window_size=11,
        epochs=100,
        batch_size=128,
        num_workers=0,
        learning_rate=0.0001,
        ablation_mode="decoupled",
    )
    if exp.random_seed not in (6, 7, 8) or any(getattr(exp, k) != v for k, v in fixed.items()):
        raise ValueError("Unapproved training protocol or seed")
    if dict(exp.paper_transfer or {}) != {"family": "SRNCF", "group": group} or exp.distillation:
        raise ValueError("Unapproved SR-NCF fusion settings")
    if dict(exp.sr or {}) != {"spatial_sigma": 3.0, "spectral_scale": 0.2775005102157593}:
        raise ValueError("Original SR2 settings required")

    net = NeighborTransMixer(
        "SR2",
        spectral_scale=exp.sr["spectral_scale"],
        spatial_sigma=3.0,
        hsi_channels=30,
        aux_channels=1,
        num_classes=15,
        hidden_dim=64,
        num_blocks=2,
        window_size=11,
        mode="decoupled",
    )
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(exp.random_seed)
        net.fusion_conv = SRNCFFusionStage(net.fusion_conv, net.gap, group, exp.random_seed, window_size=11)
    return {
        "net": net.to(device),
        "diagnostics_path": str(Path(exp.report_path).with_suffix(".diagnostics.json")),
    }


def forward_train(bundle, batch):
    return bundle["net"](batch["hsi_pca"], batch["aux"])


def forward_eval(bundle, batch):
    return forward_train(bundle, batch)


def compute_loss(bundle, outputs, batch, criterion):
    return bundle["net"].fusion_conv.record_loss(criterion(outputs, batch["label"]))


def on_training_complete(bundle):
    stage = bundle["net"].fusion_conv
    path = Path(bundle["diagnostics_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "family": "SRNCF",
                "group": stage.group,
                "seed": stage.seed,
                "training_steps": stage.step,
                "fusion_derangement_121": stage.fusion_permutation.detach().cpu().tolist(),
                "history": stage.history,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
