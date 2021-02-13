from typing import Optional, Any, Dict, List

import pytorch_lightning as pl
import torch
from copy import deepcopy
from torch.utils.data import DataLoader

from utilities.common import collate_fn
from utilities.builders import (build_lr_scheduler_from_cfg, build_optimizer_from_cfg, build_backbone_from_cfg,
                                build_head_from_cfg, build_loss_from_cfg, build_transform_from_cfg,
                                build_dataset_from_cfg, build_metric_from_cfg, build_loss_head_from_cfg)

CfgT = Dict[str, Any]

__all__ = ['LightningEquipmentDetNet']


class LightningEquipmentDetNet(pl.LightningModule):
    def __init__(self,
                 load_from_checkpoint: str,
                 train_stage: str,
                 backbone_cfg: CfgT = dict(),
                 loss_head_cfg: Optional[CfgT] = None,
                 metric_cfgs: List[CfgT] = list(),
                 train_transforms_cfg: CfgT = dict(),
                 val_transforms_cfg: CfgT = dict(),
                 train_dataset_cfg: CfgT = dict(),
                 val_dataset_cfg: CfgT = dict(),
                 train_dataloader_cfg: CfgT = dict(),
                 val_dataloader_cfg: CfgT = dict(),
                 optimizer_cfg: CfgT = dict(),
                 scheduler_cfg: CfgT = dict(),
                 scheduler_update_params: CfgT = dict()):
        super(LightningEquipmentDetNet, self).__init__()
        self.train_stage = train_stage
        self.backbone_cfg = backbone_cfg
        self.loss_head_cfg = loss_head_cfg
        self.metric_cfgs = metric_cfgs
        self.train_transforms_cfg = train_transforms_cfg
        self.val_transforms_cfg = val_transforms_cfg
        self.train_dataset_cfg = train_dataset_cfg
        self.val_dataset_cfg = val_dataset_cfg
        self.train_dataloader_cfg = train_dataloader_cfg
        self.val_dataloader_cfg = val_dataloader_cfg
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        self.scheduler_update_params = scheduler_update_params
        self.save_hyperparameters()

        self.scheduler = None
        self._val_dataset_names = []
        self._metric_names = set()
        self._build_models()

        if load_from_checkpoint is not None:
            loaded_dict = torch.load(load_from_checkpoint)['state_dict']
            if self.train_stage == 'transfer_learning':
                loaded_dict = {k: v for k, v in loaded_dict.items()
                               if 'backbone' in k}

                for param in self.backbone.parameters():
                    param.requires_grad = False

                for name, module in self.named_modules():
                    if 'backbone' in name and \
                        (isinstance(module, torch.nn.modules.BatchNorm1d) or
                         isinstance(module, torch.nn.modules.BatchNorm2d) or
                         isinstance(module, torch.nn.modules.BatchNorm3d)):
                        module.eval()

            self.load_state_dict(loaded_dict, strict=False)

    def _build_models(self):
        self.backbone = build_backbone_from_cfg(self.backbone_cfg.copy())
        self.loss_head = build_loss_head_from_cfg(self.loss_head_cfg)
        self.metrics = []

    def forward(self, x):
        features = self.backbone(x)
        x = self.loss_head(features)

        return x

    def loss(self, predictions, gt_info):
        return self.loss_head.loss(*predictions, gt_info)

    def training_step(self, batch, batch_idx):
        images = batch['image']
        gt_info = [{key: batch[key][info_idx] for key in list(batch.keys())}
                   for info_idx in range(len(images))]

        predictions = self(images)

        losses = self.loss(predictions, gt_info)
        for loss_name, loss_value in losses.items():
            self.log(loss_name, loss_value, prog_bar=True, on_epoch=False, on_step=True, logger=True)

        self.log("lr", self.scheduler.get_last_lr()[0], prog_bar=True, on_step=True, logger=False)
        return torch.sum(torch.stack([losses[k] for k, _ in losses.items()]))

    def validation_step(self, batch, batch_idx):
        images = batch['image']
        gt_info = [{key: batch[key][info_idx] for key in list(batch.keys())}
                   for info_idx in range(len(images))]

        predicted = self(images)
        pred_bboxes = self.loss_head.get_bboxes(predicted, gt_info)

        for metric in self.metrics:
            metric[1](pred_bboxes, gt_info)

    def validation_epoch_end(self, outputs):
        for metric_name, metric in self.metrics:
            self.log(f'{metric_name}', metric, prog_bar=True, on_epoch=True, logger=True)

        # calculate avg value for all datasets
        for metric_name in list(self._metric_names):
            metrics = torch.Tensor(
                [metric.compute() for metric_name_, metric in self.metrics if metric_name_ == metric_name])
            self.log(f'{metric_name}', metrics.mean(), prog_bar=True, on_epoch=True, logger=True)

    def configure_optimizers(self):
        optimizer = build_optimizer_from_cfg(self.parameters(), self.optimizer_cfg.copy())
        self.scheduler = build_lr_scheduler_from_cfg(optimizer, self.scheduler_cfg.copy())
        lr_scheduler_info = {'scheduler': self.scheduler, **self.scheduler_update_params}
        return [optimizer], [lr_scheduler_info]

    @staticmethod
    def __create_dataloader(transforms_cfg, dataset_cfg, dataloader_cfg):
        transforms = build_transform_from_cfg(transforms_cfg.copy())
        dataset = build_dataset_from_cfg(transforms, dataset_cfg.copy())
        return torch.utils.data.DataLoader(dataset, collate_fn=collate_fn,
                                           **dataloader_cfg)

    def train_dataloader(self):
        return self.__create_dataloader(self.train_transforms_cfg, self.train_dataset_cfg, self.train_dataloader_cfg)

    def val_dataloader(self):
        if isinstance(self.val_dataset_cfg, list):
            dataloaders = [self.__create_dataloader(self.val_transforms_cfg, dataset_cfg, self.val_dataloader_cfg) for
                           dataset_cfg in self.val_dataset_cfg]
        elif isinstance(self.val_dataset_cfg, dict):
            dataloaders = [
                self.__create_dataloader(self.val_transforms_cfg, self.val_dataset_cfg, self.val_dataloader_cfg)]
        else:
            assert False, 'incorrect val_dataset_cfg'

        for metric_cfg in deepcopy(self.metric_cfgs):
            metric_name = metric_cfg.pop('name') if 'name' in metric_cfg else metric_cfg['type'].lower()
            self._metric_names.add(metric_name)
            metric_module = build_metric_from_cfg(metric_cfg.copy())
            self.metrics.append((metric_name, metric_module))

        return dataloaders
