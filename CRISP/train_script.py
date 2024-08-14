import os
import sys
# When using please temporarily change it to your path of CRISP 
# Will address this problem later
sys.path.append('/ailab/user/liuxinyuan/projects/perturb_project/finetuning') 
from pathlib import Path
import pprint
import argparse
import logging
import sklearn
import copy
from CRISP.utils import load_config
from CRISP.trainer import Trainer
import yaml
from pprint import pformat
import pickle
import torch
# from CRISP.embedding import get_celltype_embeddings, get_chemical_representation
# from CRISP.data import load_dataset_splits, custom_collate
# from CRISP.eval import evaluate

"""
The defualt usage of CRISP is: 
    python CRISP/train_script.py --config [your path of the config file] 
                                --split [the key of split type] 
                                --savedir [the path of saved model, log file, evaluation files]
                                --seed (Optional)
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--split", type=str, required=True, help="Split key for data")
    parser.add_argument("--seed", type=int, required=True, default=0, help="Seed")
    parser.add_argument("--savedir", type=str, required=True, help="Path of save model")
    parser.add_argument("--MMD",type=float,default=0.1,help="coefficient of mmd loss")

    pars_args = parser.parse_args()
    config_path = pars_args.config

    args = load_config(config_path)
    args['dataset']['split_key'] = pars_args.split
    args["model"]['seed'] = pars_args.seed
    args['training']['save_dir'] = pars_args.savedir
    args['model']['hparams']['mmd'] = pars_args.MMD

    formatted_str = pprint.pformat(args)

    log_path = args['training']['save_dir']
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logging.basicConfig(filename= f'{log_path}/log.txt', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    logging.info(f'Argument setting: {formatted_str}')
    yaml.dump(
        args, open(f"{log_path}/config.yaml", "w"), default_flow_style=False
    )

    exp = Trainer()
    exp.init_dataset(args["dataset"],seed=args["model"]['seed'])
    logging.info(f'Finish init dataset')
    exp.init_drug_embedding(embedding=args["model"]["embedding"])
    logging.info(f'Finish init drug embedding')
    # exp.init_celltype_embedding(**args["model"]["ct_embedding"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # logging.info(f'Finish init cov embedding')
    exp.init_model(
        hparams=args["model"]["hparams"],
        seed=args["model"]['seed'],
    )
    exp.load_train()
    logging.info(f'Start training')
    exp.train(**args["training"])
    logging.info(f'Finish training')