import logging
from abc import abstractmethod
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from easydict import EasyDict as edict

from src.utils.generic_utils import get_experiment_folder_path

from ..logger import Logger


class BaseTrainer:
    def __init__(self, opt: edict, model: nn.Module, continue_path: Optional[str] = None) -> None:
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.opt = opt
        self.model = model.to(self.device)
        self.batches_done = 0
        self.current_epoch = 0

        continue_path = Path(continue_path) if continue_path else None
        if continue_path and continue_path.suffix == '.pth':
            self.logging_dir = continue_path.parent.parent
            self.ckpt_name = continue_path.name
        elif continue_path:
            self.logging_dir = Path(continue_path)
            self.ckpt_name = None
        else:
            self.logging_dir = Path(get_experiment_folder_path(opt.logging_dir, opt.model.kind, opt.get('experiment_name', 'exp')))
            self.ckpt_name = None
        self.vis_dir = self.logging_dir / 'visualizations'
        self.ckpt_dir = self.logging_dir / 'checkpoints'
        self.logger = Logger(self.logging_dir)
        if continue_path is not None:
            assert self.logging_dir.exists(), f'Unable to find model directory {continue_path}'
            self.restore_state()
            print(f'loaded model from {continue_path}')
        else:
            self.vis_dir.mkdir(exist_ok=True)
            self.ckpt_dir.mkdir(exist_ok=True)
            self.logger.reset()
        logging.info(f'Using model: {self.model.__class__.__name__}')

    @abstractmethod
    def restore_state(self):
        """Restore trainer's state based on the latest checkpoint from `self.ckpt_dir`"""

    @abstractmethod
    def save_state(self) -> str:
        """Persists trainer's state to the `self.ckpt_dir`"""

    @abstractmethod
    def training_epoch(self, loader: torch.utils.data.DataLoader) -> dict:
        """Runs a training epoch for a given loader and returns the results, which can include epoch loss, metrics, etc"""

    @torch.no_grad()
    def validation_epoch(self, loader: torch.utils.data.DataLoader) -> dict:
        """Runs a validation epoch for a given loader and returns the results, which can include epoch loss, metrics, etc"""

    @abstractmethod
    def get_dataloaders(self) -> tuple[torch.utils.data.DataLoader]:
        pass

    def fit(self, wandb_logger=None):
        data_loaders = self.get_dataloaders()
        train_loader, val_loader = data_loaders
        for _ in range(self.current_epoch, self.opt.n_epochs):
            epoch_stats_train = self.training_epoch(train_loader)
            epoch_stats_val = self.validation_epoch(val_loader)
            if 'classification' in self.opt.task_name and wandb_logger is not None:
                wandb_logger.log(
                    {
                        'epoch': self.current_epoch,
                        'train/loss': epoch_stats_train['loss'].item(),
                        'train/BinaryPrecision': epoch_stats_train['BinaryPrecision'].item(),
                        'train/BinaryRecall': epoch_stats_train['BinaryRecall'].item(),
                        'train/F1': epoch_stats_train['BinaryF1Score'].item(),
                        'val/loss': epoch_stats_val['loss'].item(),
                        'val/BinaryPrecision': epoch_stats_val['BinaryPrecision'].item(),
                        'val/BinaryRecall': epoch_stats_val['BinaryRecall'].item(),
                        'val/F1': epoch_stats_val['BinaryF1Score'].item(),
                    }
                )

            elif 'counterfactual_inpainting' in self.opt.task_name and wandb_logger is not None:
                # return keys: ['g_adv', 'g_kl', 'g_rec_loss', 'g_minc_loss', 'g_tv', 'g_loss', 'd_real_loss', 'd_fake_loss', 'd_loss', 'counter_acc', 'cv_80', 'fid', 'cf_iou_xc'])
                wandb_logger.log(
                    {
                        'epoch': self.current_epoch,
                        'train/g_loss': epoch_stats_train['g_loss'],
                        'train/d_real_loss': epoch_stats_train['d_real_loss'],
                        'train/d_fake_loss': epoch_stats_train['d_fake_loss'],
                        'train/d_loss': epoch_stats_train['d_loss'],
                        'train/counter_acc': epoch_stats_train.get('counter_acc', None),
                        'train/cv_80': epoch_stats_train.get('cv_80', None),
                        'train/cf_iou_xc': epoch_stats_train.get('cf_iou_xc', None),
                        'train/cf_dice_xc': epoch_stats_train.get('cf_dice_xc', None),
                        'train/fid': epoch_stats_train.get('fid', None),
                        'val/g_loss': epoch_stats_val['g_loss'],
                        'val/d_real_loss': epoch_stats_val['d_real_loss'],
                        'val/d_fake_loss': epoch_stats_val['d_fake_loss'],
                        'val/d_loss': epoch_stats_val['d_loss'],
                        'val/counter_acc': epoch_stats_train.get('counter_acc', None),
                        'val/cv_80': epoch_stats_val.get('cv_80', None),
                        'val/cf_iou_xc': epoch_stats_val.get('cf_iou_xc', None),
                        'val/cf_dice_xc': epoch_stats_val.get('cf_dice_xc', None),
                        'val/fid': epoch_stats_val.get('fid', None),
                    }
                )

            # TODO: add checkpoint saving based on the metric monitored in epoch stats
            if self.current_epoch % self.opt.checkpoint_freq == 0:
                ckpt_path = self.save_state()
                self.logger.info(f'Saved checkpoint parameters at epoch {self.current_epoch}: {ckpt_path}')
            self.current_epoch += 1

    def fit_val(self, wandb_logger=None, genlabel1 = False):
        import csv

        import numpy as np

        data_loaders = self.get_dataloaders()
        if genlabel1:
            val_loader, _ = data_loaders #this is train_loader.
        else:
            _, val_loader = data_loaders
        if genlabel1:
            epoch_stats_val = self.test_epoch_genlabel1(val_loader)
        else:
            epoch_stats_val = self.test_epoch(val_loader)
        dice_list = epoch_stats_val['each_dice']
        median_dice = np.median(dice_list)
        mean_dice = np.mean(dice_list)

        try:
            logging.info(f'Median Dice: {median_dice}')
            logging.info(f'Mean Dice: {mean_dice}')
        except:
            print(f'Median Dice: {median_dice}')
            print(f'Mean Dice: {mean_dice}')

        file_path = '/media/NAS06/gavinyue/disentanglement/benchmark/counterfactual-search/dice_score.csv'
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Dice Value'])
            # Write the dice_list to the CSV file
            for dice_value in dice_list:
                writer.writerow([dice_value])

        print(f'dice_list has been written to {file_path}')
