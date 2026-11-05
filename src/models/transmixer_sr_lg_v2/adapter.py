"""Native adapter only; no training/data/selection/metric implementation."""
import math

from ..transmixer_net.model import TransMixerNet
from ..transmixer_lg_controls.model import replace_operators
from .model import SRBalancedTransMixer

NAMES = {"TransMixer_LGR2_GG_Avg": "global_average", "TransMixer_LGR2_Uniform": "uniform",
         "TransMixer_LGR2_SR": "aligned", "TransMixer_LGR2_Shuffled": "misaligned"}


def build_model(config, dataset_type, device):
    manager = getattr(config, "_global_config_manager", None)
    exp = manager.config if manager is not None else getattr(config, "config", config)
    if dataset_type != 0 or exp.random_seed not in (6, 7, 8):
        raise ValueError("Houston2013 seeds6/7/8 only")
    if exp.ablation_mode != "decoupled" or exp.channels != 30 or exp.window_size != 11:
        raise ValueError("Keep original fixed1:1, PCA30, window11")
    settings = dict(exp.sr or {})
    if set(settings) != {"spatial_sigma", "spectral_scale"} or any(
            not math.isfinite(float(v)) or float(v) <= 0 for v in settings.values()):
        raise ValueError("Require unchanged positive finite SR2 settings")
    if exp.model_name not in NAMES:
        raise ValueError("Unknown refactor model name")
    mode = NAMES[exp.model_name]
    kwargs = dict(hsi_channels=config.get_value("channels"),
        aux_channels=config.get_value("lidar_or_sar_channels")[dataset_type],
        num_classes=config.get_value("out_features")[dataset_type], hidden_dim=64,
        num_blocks=2, window_size=11, mode="decoupled")
    if mode == "global_average":
        # Exactly the previous GG construction, with native average pooling.
        net = replace_operators(TransMixerNet(**kwargs), "GG")
    else:
        net = SRBalancedTransMixer(mode, **settings, **kwargs)
    return {"net": net.to(device)}


def forward_train(bundle, batch):
    return bundle["net"](batch["hsi_pca"], batch["aux"])


def forward_eval(bundle, batch):
    return forward_train(bundle, batch)
