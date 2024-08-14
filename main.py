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
    path = os.path.join(args.dir, t_str)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def init_logger(run_dir, args):
    log_train_dir = os.path.join(run_dir, 'log', 'train.log')
    log_test_dir = os.path.join(run_dir, 'log', 'test.log')

    log_train = get_logger('train', log_train_dir)
    log_test = get_logger('test', log_test_dir)

    log_train.info(f'Train logging at {log_train_dir}')
    log_test.info(f'Test logging at {log_test_dir}')

    log_train.info(f'Starting Deep Learning Project')
    log_train.info(f'Current Input Parameter')
    log_train.info(args)

    log_train.info(f'File Config')
    for (k, v) in config.items():
        log_train.info(f'{k:20} -> {v}')


def run(args):
    config = OmegaConf.load(f'.\\conf\\{args.conf}.yaml')
    n_fold = config['n_fold']

    run_dir = get_unique_run_dir(args)
    reproduce(args.seed)
    init_logger(run_dir, args)

    # Fold Loop
    for fold in range(0, n_fold):
        train_fold(config, run_dir, fold)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dir', type=str, default=r'.\\runs\\utsn')
    parser.add_argument('--conf', type=str, default=r'utsn')

    args = parser.parse_args()
    run(args=args)
