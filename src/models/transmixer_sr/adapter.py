"""SR pooling attached through the lab's standard three-function model API.

Keep the original SR implementation import path valid for old whole-model
checkpoints. Neither this adapter nor that model implements data or training.
"""
import math

from ..transmixer_paper.model import NeighborTransMixer


def build_model(config, dataset_type, device):
    manager = getattr(config, "_global_config_manager", None)
    exp = manager.config if manager is not None else getattr(config, "config", config)
    if dataset_type != 0:
        raise ValueError("Current SR experiments are scoped to Houston2013")
    variants = {"TransMixer_SR1": "SR1", "TransMixer_SR2": "SR2"}
    if exp.model_name not in variants:
        raise ValueError("Use a registered TransMixer_SR1 or TransMixer_SR2 model")
    if exp.ablation_mode not in ("decoupled", "pure_ssm"):
        raise ValueError("SR supports the existing decoupled and pure_ssm modes")
    settings = dict(exp.sr or {})
    if set(settings) - {"spatial_sigma", "spectral_scale"}:
        raise ValueError("Unknown SR model settings")
    variant = variants[exp.model_name]
    sigma = float(settings.get("spatial_sigma", 3.))
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("spatial_sigma must be finite and positive")
    if variant == "SR2" and "spectral_scale" not in settings:
        raise ValueError("SR2 requires spectral_scale measured from native training inputs")
    scale = float(settings.get("spectral_scale", 1.))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("spectral_scale must be finite and positive")
    net = NeighborTransMixer(
        variant, spectral_scale=scale, spatial_sigma=sigma,
        hsi_channels=config.get_value("channels"),
        aux_channels=config.get_value("lidar_or_sar_channels")[dataset_type],
        num_classes=config.get_value("out_features")[dataset_type],
        hidden_dim=64, num_blocks=2, window_size=config.get_value("windowSize"),
        mode=exp.ablation_mode,
    ).to(device)
    return {"net": net}


def forward_train(bundle, batch):
    return bundle["net"](batch["hsi_pca"], batch["aux"])


def forward_eval(bundle, batch):
    return forward_train(bundle, batch)
