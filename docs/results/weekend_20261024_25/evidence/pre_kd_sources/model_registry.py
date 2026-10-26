from importlib import import_module


# Canonical names are used by new experiments and by train_all. Legacy names
# remain accepted through MODEL_ALIASES without creating duplicate jobs.
MODEL_REGISTRY = {
    **{f"TransMixer_Fusion_F{i}": "src.models.transmixer_fusion.adapter" for i in range(7)},
    "TransMixer_B0_Decoupled": "src.models.transmixer_net.adapter",
    "TransMixer_B1_LocalResidual": "src.models.transmixer_ablation.adapter",
    "TransMixer_B2_PreSSMResidual": "src.models.transmixer_ablation.adapter",
    "TransMixer_B3_DeltaInit": "src.models.transmixer_ablation.adapter",
    "TransMixer_B4_MultiScaleDW": "src.models.transmixer_ablation.adapter",
    "TransMixer_DynamicSplit": "src.models.transmixer_net.adapter",
    "TransMixer_SoftChannelGate": "src.models.transmixer_net.adapter",
    "TransMixer_CrossStreamGate": "src.models.transmixer_net.adapter",
    "TransMixer_HybridGate": "src.models.transmixer_net.adapter",
    "TransMixer_LocalConv": "src.models.transmixer_net.adapter",
    "TransMixer_GlobalAttention": "src.models.transmixer_net.adapter",
    "TransMixer_PureSSM": "src.models.transmixer_net.adapter",
    "TransMixer_DEG": "src.models.transmixer_deg.adapter",
    "TransMixer_DEGOnly": "src.models.transmixer_deg.adapter",
    "FusAtNet": "src.models.fusatnet.adapter",
    "ExViT": "src.models.exvit.adapter",
    "HybridSN": "src.models.hybridsn.adapter",
    "MICF_Net": "src.models.micf_net.adapter",
    "CNN": "src.models.cnn.adapter",
    "FDNet": "src.models.fdnet.adapter",
    "S2ENet": "src.models.s2enet.adapter",
    "AsyFFNet": "src.models.asyffnet.adapter",
    "HCTNet": "src.models.hctnet.adapter",
    "MACN": "src.models.macn.adapter",
    "RSCNet": "src.models.rscnet.adapter",
    "DFFNet": "src.models.dffnet.adapter",
    "SGFNet": "src.models.sgfnet.adapter",
}


MODEL_ALIASES = {
    "TransMixerNet": "TransMixer_B0_Decoupled",
    "TransMixer_B1": "TransMixer_B1_LocalResidual",
    "TransMixer_B2": "TransMixer_B2_PreSSMResidual",
    "TransMixer_B3": "TransMixer_B3_DeltaInit",
    "TransMixer_B4": "TransMixer_B4_MultiScaleDW",
    "TransMixer_Dyn": "TransMixer_DynamicSplit",
    "TransMixer_Soft": "TransMixer_SoftChannelGate",
    "TransMixer_Gate": "TransMixer_CrossStreamGate",
    "TransMixer_DynGate": "TransMixer_HybridGate",
    "TransMixer_Conv": "TransMixer_LocalConv",
    "TransMixer_Attn": "TransMixer_GlobalAttention",
    "TransMixer_SSM": "TransMixer_PureSSM",
    "DEGConvNet": "TransMixer_DEGOnly",
    "TransMixer_DEGConv": "TransMixer_DEG",
}


def get_model_adapter(model_name):
    canonical_name = MODEL_ALIASES.get(model_name, model_name)
    if canonical_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    return import_module(MODEL_REGISTRY[canonical_name])
