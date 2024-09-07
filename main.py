import os
import random
import datetime

import torch
import numpy as np
from argparse import ArgumentParser
from omegaconf import OmegaConf

from train import train_fold
from utils.logger import get_logger


def reproduce(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_unique_run_dir(args):
    t_str = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    path = os.path.join(args.dir, args.conf, t_str)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def init_logger(cfg, run_dir, args):
    log_train_dir = os.path.join(run_dir, 'log', 'train.log')
    log_test_dir = os.path.join(run_dir, 'log', 'test.log')

    log_train = get_logger('train', log_train_dir)
    log_test = get_logger('test', log_test_dir)

    log_train.info(f'Train logging at {log_train_dir}')
    log_test.info(f'Test logging at {log_test_dir}')

    log_train.info(f'Starting Deep Learning Project')
    log_train.info(f'Current Input Parameter')
    log_train.info(args)

    log_train.info(f'Config: ' + f'{args.conf}_{args.data}.yaml')
    log_train.info(cfg)
    # for (k, v) in cfg.items():
    #     log_train.info(f'{k:20} -> {v}')


def run(args):
    cfg = OmegaConf.load(f'.\\conf\\{args.conf}_{args.data}.yaml')
    n_fold = cfg.train.n_fold

    run_dir = get_unique_run_dir(args)
    reproduce(args.seed)

    init_logger(cfg, run_dir, args)

    # Fold Loop
    for fold in range(0, n_fold):
        train_fold(cfg, run_dir, fold)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=255)
    parser.add_argument('--dir', type=str, default=r'.\\runs')
    parser.add_argument('--conf', type=str, default=r'attn', choices=['utsn', 'tiny', 'attn'])
    parser.add_argument('--data', type=str, default=r'mice', choices=['mice', 'sleepedf'])

    args = parser.parse_args()
    run(args=args)
