from .model import FeatureAblationNet


MODEL_TO_VARIANT = {
    "TransMixer_B1_LocalResidual": "B1",
    "TransMixer_B2_PreSSMResidual": "B2",
    "TransMixer_B3_DeltaInit": "B3",
    "TransMixer_B4_MultiScaleDW": "B4",
    "TransMixer_B1": "B1",
    "TransMixer_B2": "B2",
    "TransMixer_B3": "B3",
    "TransMixer_B4": "B4",
}


def build_model(config, dataset_type, device):
    if getattr(config, "_global_config_manager", None) is not None:
        exp = config._global_config_manager.config
    else:
        exp = config.config
    if exp.ablation_mode != "decoupled":
        raise ValueError("B1-B4 require ablation_mode='decoupled' (fixed 1:1).")
    variant = MODEL_TO_VARIANT.get(exp.model_name)
    if variant is None:
        raise ValueError(f"Unknown TransMixer feature variant: {exp.model_name}")
    net = FeatureAblationNet(
        variant=variant,
        hsi_channels=config.get_value("channels"),
        aux_channels=config.get_value("lidar_or_sar_channels")[dataset_type],
        num_classes=config.get_value("out_features")[dataset_type],
        hidden_dim=64,
        num_blocks=2,
        window_size=config.get_value("windowSize"),
    ).to(device)
    return {"net": net}


def forward_train(bundle, batch):
    return bundle["net"](batch["hsi_pca"], batch["aux"])


def forward_eval(bundle, batch):
    return forward_train(bundle, batch)
