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
import pandas as pd

logging.basicConfig(level=logging.INFO)

# Constants
# GPU_NUM = '1'
# # Set GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = GPU_NUM

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
    # select the split of dataset
    split = 'test'
    opt.dataset[split] = True
    opt.dataset.datasets[0].scan_params['load_masks'] = False
    
    # Input:{
    # opt.dataset.datasets[0]['kind'] = 'fibrosis' #'AustraliaDataset'
    # save_base_dir = '/media/NAS06/gavinyue/disentanglement/benchmark/counterfactual-search/australia_seg_results'
    # load_base_dir = '/media/NAS06/gavinyue/disentanglement/scripts_segmentation/unet_train_test/quantification_result'
    
    save_base_dir = '../oisc_seg_results'
    load_base_dir = '/media/NAS06/gavinyue/genai-wsss/data/OSIC/processed/fibrosis'
    data_split_dir = '/media/NAS06/gavinyue/genai-wsss/data/OSIC/split/splits_orig_name'
    # }

    # # data loading
    # # Load test set case IDs from CSV
    # df = pd.read_csv('/media/NAS06/gavinyue/genai-wsss/data/OSIC/doctor_category.csv')
    # caseids_test = df['case_id'][df['test'] == 1].tolist()
    # caseids_test = [str(caseid).zfill(3) for caseid in caseids_test]
    
        
    opt.dataset.datasets[0]['root_dir'] = load_base_dir
    opt.dataset.datasets[0]['split_dir'] = data_split_dir
    
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
    trainer.save_dir = trainer.save_base_dir
    os.makedirs(trainer.save_base_dir, exist_ok=True)
    os.makedirs(trainer.save_dir, exist_ok=True)
    
    trainer.infer(wandb_logger=expriment)
    logging.info(f'Finished infer for {split} cases')
        
        
    return 1
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
    This script is used for model inference on the OSIC fibrosis dataset.
    fibrosis seg model dataset input: shape (1, 1, 256, 256), dtype float32, range [-1, 1].
    Saved segmentation results: shape (256,256), uint8, range [0, 256].
    '''
    args = parser.parse_args()
    # args.config_path = (
    #     '/media/NAS06/gavinyue/disentanglement/benchmark/counterfactual-search/configs/counterfactual/paper_experiments/tuh/cf_inpainting_fibrosis.yaml'
    # )
    args.continue_path = '/media/NAS06/gavinyue/disentanglement/benchmark/counterfactual-search/tools/training_logs/counterfactual/fibrosis/inpainting_counterfactual_cgan-August-28-2024_11+46PM-fibrosis+covid/checkpoints/checkpoint_80.pth'
    main(args)
