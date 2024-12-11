import argparse
import logging
import os
import shutil

import yaml
from easydict import EasyDict as edict

from src.models import build_model
from src.trainers import build_trainer
from src.utils.generic_utils import seed_everything

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_path', type=str, required=False, help='Configuration file path to start training from scratch')
parser.add_argument('-cp', '--continue_path', type=str, required=False, help='Path to the existing training run to continue interrupted training')
opt = parser.parse_args()


def main(args):
    with open(args.config_path or os.path.join(args.continue_path, 'hparams.yaml')) as fid:
        opt = yaml.safe_load(fid)
        opt = edict(opt)
    seed_everything(opt.seed)

    model = build_model(opt.task_name, opt=opt.model, img_size=opt.dataset.img_size)
    trainer = build_trainer(opt.task_name, opt, model, args.continue_path)

    if args.continue_path is None:
        shutil.copy2(args.config_path, trainer.logging_dir / 'hparams.yaml')
    logging.info('Started training.')
    trainer.fit()
    logging.info('Finished training.')


if __name__ == '__main__':
    args = parser.parse_args()
    args.config_path = '/media/NAS06/gavinyue/disentanglement/benchmark/counterfactual-search/configs/classification/fibrosis/effv2s_fibrosis_ctnorm.yaml'
    # args.config_path = "/media/NAS06/gavinyue/disentanglement/benchmark/counterfactual-search/configs/counterfactual/paper_experiments/tuh/cf_inpainting_fibrosis.yaml"
    main(args)

# device = "cuda"
# # load classification model
# model = nn.Linear(512, 2)
# model = model.to(device)
# # state_dict = torch.load('/media/NAS04/yyfang/prognostic_result/xai/counterfactual/Diffusion-Explainer/scripts_osic/result_exp/classification_fibrosis/test/model/model_loss_best.pt')['model_state_dict']
# state_dict = torch.load(cls_model_path)['model_state_dict']

# if any(key.startswith('module.') for key in state_dict.keys()):
#     print(f"Using {torch.cuda.device_count()} GPUs!")
#     model = nn.DataParallel(model)

#     model.load_state_dict(state_dict)
#     direction_class_0 = model.module.weight[0]
#     direction_class_1 = model.module.weight[1]
# else:
#     model.load_state_dict(state_dict)
#     direction_class_0 = model.weight[0]
#     direction_class_1 = model.weight[1]
# model.eval()
