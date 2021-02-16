import pretrainedmodels
import timm
import albumentations as Albumentations
import albumentations.pytorch as AlbumentationsPytorch

import pytorch_lightning.callbacks as LightningCallbacks

import torch.optim as OptimizerLib
import torch.optim.lr_scheduler as LRSchedulerLib
import torch_optimizer as OptimSecondLib

import pipeline.datasets as Datasets
import pipeline.losses as Losses
import nn_constructor.loss_heads as LossHeads
import pipeline.metrics as CustomMetrics
import nn_constructor.backbones as Backbones
import nn_constructor.heads as Heads
import pipeline.transforms as Transforms

import torch
from typing import Tuple

__all__ = [
    'build_lightning_module',
    'build_backbone_from_cfg',
    'build_head_from_cfg',
    'build_transform_from_cfg',
    'build_dataset_from_cfg',
    'build_loss_head_from_cfg',
    'build_loss_from_cfg',
    'build_metric_from_cfg',
    'build_optimizer_from_cfg',
    'build_lr_scheduler_from_cfg',
    'build_callbacks_from_cfg',
    'build_loss_head_from_cfg'
]


def _base_transform_from_cfg(config, modules_to_find):
    assert isinstance(config, dict) and 'type' in config, f'Check config type validity: {config}'

    args = config.copy()
    obj_type_name = args.pop('type')

    real_type = None
    for module in modules_to_find:
        if not hasattr(module, obj_type_name):
            continue
        real_type = getattr(module, obj_type_name)

    assert real_type is not None, f'{obj_type_name} is not registered type in any modules: {modules_to_find}'
    return real_type(**args)


def build_lightning_module(config):
    import modules as Modules
    return _base_transform_from_cfg(config, [Modules])


def build_backbone_from_cfg(config) -> Tuple[torch.nn.Module, int]:
    args = config.copy()
    backbone_type_name = args.pop('type')

    if hasattr(Backbones, backbone_type_name):
        backbone = getattr(Backbones, backbone_type_name)(**args)
    elif backbone_type_name in pretrainedmodels.__dict__:
        backbone = pretrainedmodels.__dict__[backbone_type_name](**args)
        backbone.forward = backbone.features
    elif backbone_type_name in timm.list_models():
        backbone = timm.create_model(backbone_type_name, **args)
        backbone.forward = backbone.forward_features
    else:
        assert False, f'{backbone_type_name} not found in backbones factory'

    return backbone


def build_head_from_cfg(input_channels: int, config):
    config['input_channels'] = input_channels
    return _base_transform_from_cfg(config, [Heads])


def build_loss_head_from_cfg(config):
    return _base_transform_from_cfg(config, [LossHeads])


def build_transform_from_cfg(config):
    def _builder(cfg):
        if 'transforms' in cfg:
            cfg['transforms'] = [
                _builder(transform_cfg) for transform_cfg in cfg['transforms']
            ]

        return _base_transform_from_cfg(cfg, [Albumentations, AlbumentationsPytorch,
                                              Transforms])

    return _builder(config)


def build_dataset_from_cfg(transforms, config):
    config['transforms'] = transforms
    return _base_transform_from_cfg(config, [Datasets])


def build_loss_from_cfg(config):
    return _base_transform_from_cfg(config, [Losses])


def build_metric_from_cfg(config):
    return _base_transform_from_cfg(config, [CustomMetrics])


def build_optimizer_from_cfg(params, config):
    modules = [OptimizerLib, OptimSecondLib]
    try:
        import adabelief_pytorch
        modules.append(adabelief_pytorch)
    except ImportError:
        pass

    try:
        import ranger_adabelief
        modules.append(ranger_adabelief)
    except ImportError:
        pass

    try:
        import ranger
        modules.append(ranger)
    except ImportError:
        pass

    try:
        import nn_constructor.custom_optimizers as opts
        modules.append(opts)
    except ImportError:
        pass

    config['params'] = params
    return _base_transform_from_cfg(config, modules)


def build_lr_scheduler_from_cfg(optimizer, config):
    config['optimizer'] = optimizer
    return _base_transform_from_cfg(config, [LRSchedulerLib])


def build_callbacks_from_cfg(config):
    return _base_transform_from_cfg(config, [LightningCallbacks])

