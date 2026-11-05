"""Original SR2 network plus one operator substitution; native training API."""
from ..transmixer_paper.model import NeighborTransMixer
from .model import replace_operators


def build_model(config, dataset_type, device):
    manager = getattr(config, '_global_config_manager', None)
    exp = manager.config if manager is not None else getattr(config, 'config', config)
    if dataset_type != 0 or exp.random_seed not in (6, 7, 8):
        raise ValueError('Approved Houston2013 seeds6/7/8 only')
    names = {'TransMixer_LG_LL_SR2':'LL', 'TransMixer_LG_GG_SR2':'GG'}
    kind = names[exp.model_name]
    settings = dict(exp.sr or {})
    if set(settings) != {'spatial_sigma', 'spectral_scale'}:
        raise ValueError('Require unchanged SR2 settings')
    net = NeighborTransMixer('SR2', spectral_scale=settings['spectral_scale'],
        spatial_sigma=settings['spatial_sigma'], hsi_channels=config.get_value('channels'),
        aux_channels=config.get_value('lidar_or_sar_channels')[dataset_type],
        num_classes=config.get_value('out_features')[dataset_type], hidden_dim=64,
        num_blocks=2, window_size=config.get_value('windowSize'), mode='decoupled')
    return {'net':replace_operators(net, kind).to(device)}


def forward_train(bundle, batch):
    return bundle['net'](batch['hsi_pca'], batch['aux'])


def forward_eval(bundle, batch):
    return forward_train(bundle, batch)
