from .model import FusionTransMixerNet, MODEL_NAMES


def build_model(config, dataset_type, device):
    manager = getattr(config, "_global_config_manager", None)
    cfg = manager.config if manager is not None else getattr(config, "config", config)
    name = cfg.model_name
    reverse = {value: key for key, value in MODEL_NAMES.items()}
    if name not in reverse:
        raise ValueError(f"Unknown fusion model: {name}")
    net = FusionTransMixerNet(
        reverse[name], hsi_channels=config.get_value("channels"),
        aux_channels=config.get_value("lidar_or_sar_channels")[dataset_type],
        num_classes=config.get_value("out_features")[dataset_type],
        window_size=config.get_value("windowSize"),
    ).to(device)
    return {"net": net}


def forward_train(bundle, batch):
    return bundle["net"](batch["hsi_pca"], batch["aux"])


def forward_eval(bundle, batch):
    return forward_train(bundle, batch)
