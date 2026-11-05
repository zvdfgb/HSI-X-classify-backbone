from importlib import import_module


# Only current models appear in CLI choices and the native train_all queue.
# Historical adapters stay addressable for explicit old configurations.
MODEL_REGISTRY = {
    **{f"TransMixer_SRP_P{i:02d}": "src.models.transmixer_sr_prior.adapter" for i in (1, 2)},
    **{f"TransMixer_FV2_{family}{i:02d}": "src.models.transmixer_fusion_v2.adapter"
       for family in "FUD" for i in range(1, 5)},
    **{f"TransMixer_FP_G{i:02d}": "src.models.transmixer_fusion_papers.adapter"
       for i in range(2, 15)},
    "TransMixer_SR1": "src.models.transmixer_sr.adapter",
    "TransMixer_SR2": "src.models.transmixer_sr.adapter",
    "TransMixer_B0_Decoupled": "src.models.transmixer_net.adapter",
    "TransMixer_DynamicSplit": "src.models.transmixer_net.adapter",
    "TransMixer_SoftChannelGate": "src.models.transmixer_net.adapter",
    "TransMixer_CrossStreamGate": "src.models.transmixer_net.adapter",
    "TransMixer_HybridGate": "src.models.transmixer_net.adapter",
    "TransMixer_LocalConv": "src.models.transmixer_net.adapter",
    "TransMixer_GlobalAttention": "src.models.transmixer_net.adapter",
    "TransMixer_PureSSM": "src.models.transmixer_net.adapter",
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


LEGACY_MODEL_REGISTRY = {
    **{f"TransMixer_Paper_{key}": "src.models.transmixer_paper.adapter"
       for key in ("SR1", "SR2", "SR3", "PN1", "PN2", "OC1", "OC2", "OC3")},
    "TransMixer_KD_B0_RSC": "src.models.transmixer_distill.adapter",
    "TransMixer_KD_F4_RSC": "src.models.transmixer_distill.adapter",
    "TransMixer_KD_F4_CNN": "src.models.transmixer_distill.adapter",
    "TransMixer_KD_F4_RSC_Confidence": "src.models.transmixer_distill.adapter",
    "TransMixer_Control_F4_LabelSmooth": "src.models.transmixer_distill.adapter",
    **{f"TransMixer_Fusion_F{i}": "src.models.transmixer_fusion.adapter" for i in range(7)},
    "TransMixer_B1_LocalResidual": "src.models.transmixer_ablation.adapter",
    "TransMixer_B2_PreSSMResidual": "src.models.transmixer_ablation.adapter",
    "TransMixer_B3_DeltaInit": "src.models.transmixer_ablation.adapter",
    "TransMixer_B4_MultiScaleDW": "src.models.transmixer_ablation.adapter",
    "TransMixer_DEG": "src.models.transmixer_deg.adapter",
    "TransMixer_DEGOnly": "src.models.transmixer_deg.adapter",
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
    module = MODEL_REGISTRY.get(canonical_name) or LEGACY_MODEL_REGISTRY.get(canonical_name)
    if module is None:
        raise ValueError(f"Unknown model: {model_name}")
    return import_module(module)
