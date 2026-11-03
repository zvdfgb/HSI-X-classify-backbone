"""Native adapter; audited v2 matrix only. No training protocol replacement."""
from ..transmixer_paper.model import NeighborTransMixer
from .model import FusionStage, GROUPS


def build_model(config, dataset_type, device):
    manager = getattr(config, "_global_config_manager", None)
    exp = manager.config if manager is not None else getattr(config, "config", config)
    group = exp.model_name.removeprefix("TransMixer_FV2_")
    if group not in GROUPS or exp.model_name != f"TransMixer_FV2_{group}" or dataset_type != 0:
        raise ValueError("Only approved Houston2013 FV2 groups")
    fixed = dict(channels=30, window_size=11, epochs=100,
                 batch_size=128, num_workers=0, learning_rate=0.0001, ablation_mode="decoupled")
    if exp.random_seed not in (6, 7, 8) or any(getattr(exp, k) != v for k, v in fixed.items()):
        raise ValueError("Unapproved FV2 training protocol")
    if dict(exp.paper_transfer or {}) != {"group": group} or exp.distillation:
        raise ValueError("Only explicit approved FV2 group settings")
    if dict(exp.sr or {}) != {"spatial_sigma": 3.0, "spectral_scale": 0.2775005102157593}:
        raise ValueError("Original training-derived SR2 settings required")
    net = NeighborTransMixer("SR2", spectral_scale=exp.sr["spectral_scale"], spatial_sigma=3.0,
        hsi_channels=config.get_value("channels"),
        aux_channels=config.get_value("lidar_or_sar_channels")[dataset_type],
        num_classes=config.get_value("out_features")[dataset_type],
        hidden_dim=64, num_blocks=2, window_size=11, mode="decoupled")
    net.fusion_conv = FusionStage(net.fusion_conv, group)
    if hasattr(net.fusion_conv, "compensation"):
        net.hsi_proj.register_forward_hook(net.fusion_conv.capture_h)
        net.aux_stem.register_forward_hook(net.fusion_conv.capture_l)
    return {"net": net.to(device)}


def forward_train(bundle, batch):
    return bundle["net"](batch["hsi_pca"], batch["aux"])


def forward_eval(bundle, batch):
    return forward_train(bundle, batch)


def compute_loss(bundle, outputs, batch, criterion):
    net = bundle["net"]
    return net.fusion_conv.additional_loss(criterion(outputs, batch["label"]), criterion, batch["label"], net.gap.weights)
