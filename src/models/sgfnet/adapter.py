import torch
from .model import SGFNet

MODEL_NAME = "SGFNet"


def build_model(config, dataset_type, device):
    in_channels = config.get_value("channels")
    aux_channels = config.get_value("lidar_or_sar_channels")[dataset_type]
    num_classes = config.get_value("out_features")[dataset_type]
    window_size = config.get_value("windowSize")

    net = SGFNet(
        hsi_channels=in_channels,
        aux_channels=aux_channels,
        num_classes=num_classes,
        window_size=window_size,
    )
    return {"net": net}


def forward_train(bundle, batch):
    return bundle["net"](batch["hsi_pca"], batch["aux"])


def forward_eval(bundle, batch):
    return bundle["net"](batch["hsi_pca"], batch["aux"])
