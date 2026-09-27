from .model import TransMixerNet

MODEL_NAME = "TransMixerNet"

CANONICAL_MODES = {
    "TransMixer_B0_Decoupled": "decoupled",
    "TransMixer_DynamicSplit": "dynamic",
    "TransMixer_SoftChannelGate": "soft",
    "TransMixer_CrossStreamGate": "gate",
    "TransMixer_HybridGate": "dyngate",
    "TransMixer_LocalConv": "only_conv",
    "TransMixer_GlobalAttention": "only_attn",
    "TransMixer_PureSSM": "pure_ssm",
}


def build_model(config, dataset_type, device):
    """
    读取 config 对应数据集维度，完成模型实例构建
    """
    in_channels = config.get_value("channels")
    aux_channels = config.get_value("lidar_or_sar_channels")[dataset_type]
    num_classes = config.get_value("out_features")[dataset_type]
    window_size = config.get_value("windowSize")

    # 优先从 config 读取 ablation_mode；若未指定则依据模型名推断
    exp_cfg = None
    if hasattr(config, "_global_config_manager") and config._global_config_manager is not None:
        exp_cfg = config._global_config_manager.config
    elif hasattr(config, "config"):
        exp_cfg = config.config
    elif hasattr(config, "model_name"):
        exp_cfg = config

    mode = getattr(exp_cfg, "ablation_mode", None) if exp_cfg else None
    if not mode or mode == "decoupled":
        model_name = getattr(exp_cfg, "model_name", "TransMixerNet") if exp_cfg else "TransMixerNet"
        if model_name in CANONICAL_MODES:
            mode = CANONICAL_MODES[model_name]
        else:
            name_lower = model_name.lower()
            if "dyngate" in name_lower:
                mode = "dyngate"
            elif "soft" in name_lower:
                mode = "soft"
            elif "gate" in name_lower:
                mode = "gate"
            elif "dyn" in name_lower:
                mode = "dynamic"
            elif "conv" in name_lower:
                mode = "only_conv"
            elif "attn" in name_lower or "attention" in name_lower:
                mode = "only_attn"
            elif "ssm" in name_lower:
                mode = "pure_ssm"
            else:
                mode = "decoupled"

    net = TransMixerNet(
        hsi_channels=in_channels,
        aux_channels=aux_channels,
        num_classes=num_classes,
        hidden_dim=64,
        num_blocks=2,
        window_size=window_size,
        mode=mode
    ).to(device)

    return {"net": net}


def forward_train(bundle, batch):
    """
    训练前向传播，统一返回分类 logits
    """
    return bundle["net"](batch["hsi_pca"], batch["aux"])


def forward_eval(bundle, batch):
    """
    评测/推理前向传播
    """
    return bundle["net"](batch["hsi_pca"], batch["aux"])
