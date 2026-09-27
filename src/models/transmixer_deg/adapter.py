from .model import TransMixerDEGNet

MODEL_NAME = "TransMixer_DEG"


def build_model(config, dataset_type, device):
    in_channels = config.get_value("channels")
    aux_channels = config.get_value("lidar_or_sar_channels")[dataset_type]
    num_classes = config.get_value("out_features")[dataset_type]
    window_size = config.get_value("windowSize")

    # 获取当前配置与模型名
    exp_cfg = None
    if hasattr(config, "_global_config_manager") and config._global_config_manager is not None:
        exp_cfg = config._global_config_manager.config
    elif hasattr(config, "config"):
        exp_cfg = config.config
    elif hasattr(config, "model_name"):
        exp_cfg = config

    model_name = getattr(exp_cfg, "model_name", "TransMixer_DEG") if exp_cfg else "TransMixer_DEG"
    name_lower = str(model_name).lower()

    # 区分单独 B 还是 A+B:
    # DEGConvNet -> 纯 DEGConv 各向异性条形门控主干 (单独 B)
    # TransMixer_DEG -> TransMixer 解耦 + 局部流嵌入 DEGConv (A + B)
    if "degconvnet" in name_lower or "deg_only" in name_lower or "degonly" in name_lower:
        arch = "deg_only"
    else:
        arch = "transmixer_deg"

    net = TransMixerDEGNet(
        hsi_channels=in_channels,
        aux_channels=aux_channels,
        num_classes=num_classes,
        hidden_dim=64,
        num_blocks=2,
        window_size=window_size,
        arch=arch
    ).to(device)

    return {"net": net}


def forward_train(bundle, batch):
    return bundle["net"](batch["hsi_pca"], batch["aux"])


def forward_eval(bundle, batch):
    return bundle["net"](batch["hsi_pca"], batch["aux"])

