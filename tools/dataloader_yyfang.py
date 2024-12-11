import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.datasets import get_dataloaders
from src.datasets.augmentations import get_transforms
from src.utils.generic_utils import seed_everything
from src.visualizations import CLASS_COLORS, DEFAULT_COLOR

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_path', type=str, required=True, help='Configuration file path to start training from scratch')

parser.add_argument('-st', '--step_size', type=int, default=10, help='Every nth record to be visualized')
parser.add_argument('-s', '--sampling_only', action='store_true', default=False, help='Visualize only dataset examples where the sampler ')
parser.add_argument('-na', '--no_aug', action='store_true', default=False, help='Disable augmentations for visualization')
parser.add_argument('-la', '--label_agg_order', nargs='+', type=int, help='The order at which to overlay the classes masks.')
args_input = ['-c', 'xxx']
args = parser.parse_args(args_input)
args.config_path = '/media/NAS06/gavinyue/disentanglement/benchmark/counterfactual-search/configs/classification/kits_tumours/effv2s_kits_tumors.yaml'
args.output_dir = '/media/NAS06/gavinyue/disentanglement/benchmark/counterfactual-search/kits23/slice_dataset'
args.step_size = 1


with open(args.config_path) as fid:
    opt = yaml.safe_load(fid)
    opt = edict(opt)
seed_everything(opt.seed)

print('Creating data loaders to be visualized...')
transforms = get_transforms(opt.dataset)
if args.no_aug:
    transforms['train'] = transforms['val']
params = edict(opt.dataset, use_sampler=False, reset_sampler=False, shuffle_test=False)
train_loader, test_loader = get_dataloaders(params, transforms)
print(f'Total number of samples: {len(train_loader.dataset) + len(test_loader.dataset)}')

print('saveing slice train and val set')
