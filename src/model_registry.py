from importlib import import_module


MODEL_REGISTRY = {
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
    "TransMixerNet": "src.models.transmixer_net.adapter",
    "TransMixer_Conv": "src.models.transmixer_net.adapter",
    "TransMixer_Attn": "src.models.transmixer_net.adapter",
    "TransMixer_SSM": "src.models.transmixer_net.adapter",
    "DEGConvNet": "src.models.transmixer_deg.adapter",
    "TransMixer_DEG": "src.models.transmixer_deg.adapter",
}


def get_model_adapter(model_name):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    return import_module(MODEL_REGISTRY[model_name])
