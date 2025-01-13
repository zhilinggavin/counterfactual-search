import argparse
import logging
import os
import shutil

import wandb
import yaml
from easydict import EasyDict as edict

from src.models import build_model
from src.trainers import build_trainer
from src.utils.generic_utils import seed_everything
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_path', type=str, required=False, help='Configuration file path to start training from scratch')
parser.add_argument('-cp', '--continue_path', type=str, required=False, help='Path to the existing training run to continue interrupted training')
opt = parser.parse_args()


def main(args):
    with open(args.config_path or os.path.join(args.continue_path.split('/checkpoints')[0], 'hparams.yaml')) as fid:
        opt = yaml.safe_load(fid)
        opt = edict(opt)
    seed_everything(opt.seed)


    opt.dataset['batch_size'] = 1
    opt.dataset['num_workers'] = 0
    model = None
    
    '''
    Define parameters
    '''
    opt.dataset['test'] = True
    opt.dataset.datasets[0].scan_params['load_masks'] = False
    
    # Input:{
    opt.dataset.datasets[0]['kind'] = 'AustraliaDataset'
    save_base_dir = '/media/NAS06/gavinyue/disentanglement/benchmark/counterfactual-search/australia_seg_results'
    load_base_dir = '/media/NAS06/gavinyue/disentanglement/scripts_segmentation/unet_train_test/quantification_result'
    # }
    case_names = os.listdir(load_base_dir)
    case_names = [name for name in case_names if ('_0' in name and len(name) == 8)]
    sorted(case_names)
    
    for case_name in tqdm(case_names):
        # case_name = '02_00019'
        opt.dataset.datasets[0]['root_dir'] = os.path.join(load_base_dir,case_name)
        
        if model is None:
            model = build_model(opt.task_name, opt=opt.model, img_size=opt.dataset.img_size)
            trainer = build_trainer(opt.task_name, opt, model, args.continue_path)
        
        trainer.opt = opt

        if args.continue_path is None:
            shutil.copy2(args.config_path, trainer.logging_dir / 'hparams.yaml')
        logging.info('Started training.')

        expriment = None
        # wandb.init(project="COIN", name="cf_inpainting_fibrosis", config=opt)
        # expriment = wandb.run
        

        trainer.save_base_dir = save_base_dir
        trainer.save_dir = os.path.join(trainer.save_base_dir, case_name)
        os.makedirs(trainer.save_base_dir, exist_ok=True)
        os.makedirs(trainer.save_dir, exist_ok=True)
        
        trainer.infer(wandb_logger=expriment)
        logging.info(f'Finished infer for case: {case_name}.')


if __name__ == '__main__':
    '''
    This script is used for model inference on the Austrilia fibrosis dataset.
    fibrosis seg model dataset input: shape (1, 1, 256, 256), dtype float32, range [-1, 1].
    Saved segmentation results: shape (256,256), uint8, range [0, 256].
    '''
    args = parser.parse_args()
    # args.config_path = (
    #     '/media/NAS06/gavinyue/disentanglement/benchmark/counterfactual-search/configs/counterfactual/paper_experiments/tuh/cf_inpainting_fibrosis.yaml'
    # )
    args.continue_path = '/media/NAS06/gavinyue/disentanglement/benchmark/counterfactual-search/tools/training_logs/counterfactual/fibrosis/inpainting_counterfactual_cgan-August-28-2024_11+46PM-fibrosis+covid/checkpoints/checkpoint_80.pth'
    main(args)
