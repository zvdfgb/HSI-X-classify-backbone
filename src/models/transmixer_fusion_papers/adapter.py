"""Native three-function API plus the already-supported optional loss hook."""
import math

from ..transmixer_net.model import TransMixerNet
from ..transmixer_paper.model import NeighborTransMixer
from .model import FusionStage, GROUPS


def build_model(config, dataset_type, device):
    manager = getattr(config, "_global_config_manager", None)
    exp = manager.config if manager is not None else getattr(config, "config", config)
    group = exp.model_name.removeprefix("TransMixer_FP_")
    if dataset_type != 0 or group not in GROUPS or exp.model_name != f"TransMixer_FP_{group}":
        raise ValueError("Only the approved Houston2013 fusion groups are supported")
    approved_seed = exp.random_seed == 6 or (
        group in ("G02", "G03", "G04", "G05") and exp.random_seed in (7, 8)
    )
    if not approved_seed or exp.channels != 30 or exp.window_size != 11:
        raise ValueError("Approved fusion protocol requires seed6 (B1 also seed7/8), PCA30, window11")
    if dict(exp.paper_transfer or {}) != {"group": group}:
        raise ValueError("Explicit matching paper_transfer group required")
    use_a = GROUPS[group][2]
    mode = "decoupled" if use_a else "pure_ssm"
    if exp.ablation_mode != mode:
        raise ValueError("Backbone mode disagrees with the approved group")
    common = dict(hsi_channels=config.get_value("channels"),
                  aux_channels=config.get_value("lidar_or_sar_channels")[dataset_type],
                  num_classes=config.get_value("out_features")[dataset_type],
                  hidden_dim=64, num_blocks=2, window_size=config.get_value("windowSize"), mode=mode)
    if use_a:
        settings = dict(exp.sr or {})
        if set(settings) != {"spatial_sigma", "spectral_scale"} or settings["spatial_sigma"] != 3.0:
            raise ValueError("Original SR2 settings required")
        scale = float(settings["spectral_scale"])
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Finite positive native training spectral scale required")
        net = NeighborTransMixer("SR2", spectral_scale=scale, spatial_sigma=3.0, **common)
    else:
        if exp.sr:
            raise ValueError("R controls must retain original mean pooling")
        net = TransMixerNet(**common)
    # Baseline is constructed in the original order. No RNG overrides or copied
    # initializations are introduced; native set_random_seed owns randomness.
    net.fusion_conv = FusionStage(net.fusion_conv, group)
    return {"net": net.to(device)}


def forward_train(bundle, batch):
    return bundle["net"](batch["hsi_pca"], batch["aux"])


def forward_eval(bundle, batch):
    return forward_train(bundle, batch)


def compute_loss(bundle, outputs, batch, criterion):
    return bundle["net"].fusion_conv.additional_loss(criterion(outputs, batch["label"]))
