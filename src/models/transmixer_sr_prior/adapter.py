"""Native three-function model API; default CE and original trainer untouched."""
from .model import SRPriorTransMixer


def build_model(config, dataset_type, device):
    manager = getattr(config, "_global_config_manager", None)
    exp = manager.config if manager is not None else getattr(config, "config", config)
    modes = {"TransMixer_SRP_P01": "aligned", "TransMixer_SRP_P02": "misaligned"}
    if exp.model_name not in modes or dataset_type != 0:
        raise ValueError("Only the two approved Houston2013 SR-prior groups are supported")
    expected = dict(channels=30, window_size=11, epochs=100, batch_size=128,
                    num_workers=0, learning_rate=0.0001, ablation_mode="decoupled")
    if any(getattr(exp, key) != value for key, value in expected.items()) or exp.random_seed not in (6, 7, 8):
        raise ValueError("Original 100-epoch protocol and approved seeds6/7/8 required")
    if exp.distillation is not None or exp.paper_transfer is not None:
        raise ValueError("No extra losses, transfer mechanisms or distillation in this batch")
    settings = dict(exp.sr or {})
    if settings != {"spatial_sigma": 3.0, "spectral_scale": 0.2775005102157593}:
        raise ValueError("Keep original training-input-derived SR2 settings")
    net = SRPriorTransMixer(modes[exp.model_name], **settings,
        hsi_channels=config.get_value("channels"),
        aux_channels=config.get_value("lidar_or_sar_channels")[dataset_type],
        num_classes=config.get_value("out_features")[dataset_type],
        hidden_dim=64, num_blocks=2, window_size=11, mode="decoupled").to(device)
    return {"net": net}


def forward_train(bundle, batch):
    return bundle["net"](batch["hsi_pca"], batch["aux"])


def forward_eval(bundle, batch):
    return forward_train(bundle, batch)
