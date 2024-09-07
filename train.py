import os
import logging

import numpy as np

from loader.seq.seq_loader import SeqLoader
from loader.seq.seq_preproc import SeqPreproc
from model.seq_kit import SeqKit

from loader.wnd.wnd_loader import WndLoader
from loader.wnd.wnd_preproc import WndPreproc
from model.wnd_kit import WndKit
from utils.sleep_cls import calc_class_loss_weight


def train_fold(cfg, run_dir, fold):

    # Log Init
    log_train = logging.getLogger('train')
    log_test = logging.getLogger('test')
    log_train.info(f'Fold {fold}')

    # Data Preparation
    log_train.info('Prepare Dataset')
    wnd_prep = WndPreproc(cfg)
    dataset = wnd_prep.preproc_fold(fold)

    # Class aware loss function weight update for attn sleep
    if cfg.model.name == 'attn':
        label = np.empty(shape=(0,))
        for i in range(3):
            label = np.concatenate((label, dataset[i][1]), axis=0)

        weight = calc_class_loss_weight(cfg, label)
        cfg.model.loss_class_weight = weight

    # Train Model Creation
    log_train.info('Creating Training Model')
    toolkit = WndKit(cfg, run_dir, fold, test=False)

    # Param Init
    n_epoch = cfg.train.n_epoch
    n_eval_span = cfg.train.eval_span
    best_acc = -1
    best_f1 = -1

    # Epoch Loop
    log_train.info('Start training epoch')
    for epoch in range(toolkit.cur_epoch(), n_epoch):
        train_iter = WndLoader(cfg, dataset[0]).iter()
        valid_iter = WndLoader(cfg, dataset[1]).iter()
        test_iter = WndLoader(cfg, dataset[2]).iter()

        train_metric = toolkit.train_epoch(train_iter)
        valid_metric = toolkit.eval_epoch(valid_iter)
        test_metric = toolkit.eval_epoch(test_iter)

        # toolkit.sche.step()
        step = train_metric['global_step']

        board = toolkit.tensorboard
        for (name, metric) in (
                ('train', train_metric),
                ('valid', valid_metric),
                ('test', test_metric),
                ('epoch', None)
        ):
            for item in ('loss', 'acc', 'f1', 'cohen'):
                tag = f'{name}/{item}'
                if metric is None:
                    val = epoch + 1
                else:
                    val = metric[f'{item}']

                board.add_scalar(tag=tag, scalar_value=val, global_step=step)

        log_train.info(
            # f'[e {epoch+1}/{n_epoch} | s {step} | lr {train_metric["lr"]:.3f}] '
            f'[e {epoch+1}/{n_epoch} | s {step}] '
            f'TR n={len(train_metric["truth"])} '
            f'l={train_metric["loss"]:.3f} '
            f'a={train_metric["acc"]:.3f} '
            f'f1={train_metric["f1"]:.3f} '
            f't={train_metric["dur"]:.3f} | '

            f'VA n={len(valid_metric["truth"])} '
            f'l={valid_metric["loss"]:.3f} '
            f'a={valid_metric["acc"]:.3f} '
            f'f1={valid_metric["f1"]:.3f} '
            f't={valid_metric["dur"]:.3f} | '

            f'TE n={len(test_metric["truth"])} '
            f'l={test_metric["loss"]:.3f} '
            f'a={test_metric["acc"]:.3f} '
            f'f1={test_metric["f1"]:.3f} '
            f't={test_metric["dur"]:.3f}'
        )

        if best_acc < valid_metric['acc'] and best_f1 < valid_metric['f1']:
            best_acc = valid_metric['acc']
            best_f1 = valid_metric['f1']
            toolkit.save_best_ckpt('best')

        if (epoch + 1) % n_eval_span == 0 or (epoch + 1) == n_epoch:
            # log_test.info(f'[epoch {epoch + 1}/{n_epoch} | step {step} | lr {test_metric["lr"]:.3f}]')
            log_test.info(f'[epoch {epoch + 1}/{n_epoch} | step {step}]')
            log_test.info('Train Confusion Matrix')
            log_test.info('\n' + str(train_metric['cm']))
            log_test.info('Test Confusion Matrix')
            log_test.info('\n' + str(valid_metric['cm']))
            log_test.info('\n' + str(test_metric['cm']))
