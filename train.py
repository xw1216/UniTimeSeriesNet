import os
import logging

from loader.seq_loader import SeqLoader
from loader.seq_preproc import SeqPreproc
from model.seq_kit import SeqKit


def train_fold(config, run_dir, fold):
    # Path Creation
    fold_dir = os.path.join(run_dir, str(fold))
    if not os.path.exists(fold_dir):
        os.makedirs(fold_dir)

    # Log Init
    log_train = logging.getLogger('train')
    log_test = logging.getLogger('test')
    log_train.info(f'Fold {fold}')

    # Data Preparation
    log_train.info('Prepare Dataset')
    seq_prep = SeqPreproc(config)
    dataset = seq_prep.preproc_seq(fold)

    # Train Model Creation
    log_train.info('Creating Training Model')
    toolkit = SeqKit(config, fold_dir, fold, test=False, use_best=False)

    # Param Init
    n_epoch = config['n_epoch']
    n_eval_span = config['eval_span']
    best_acc = -1
    best_f1 = -1

    # Epoch Loop
    log_train.info('Start training epoch')
    for epoch in range(toolkit.cur_epoch(), n_epoch):
        train_iter = SeqLoader(config, dataset[0]).iter()
        valid_iter = SeqLoader(config, dataset[1]).iter()
        test_iter = SeqLoader(config, dataset[2]).iter()

        train_metric = toolkit.train_epoch(train_iter)
        valid_metric = toolkit.eval_epoch(valid_iter)
        test_metric = toolkit.eval_epoch(test_iter)

        step = train_metric['global_step']

        board = toolkit.tensorboard
        for (name, metric) in (
                ('train', train_metric),
                ('valid', valid_metric),
                ('test', test_metric),
                ('epoch', None)
        ):
            for item in ('loss', 'acc', 'f1'):
                tag = f'{name}/{item}'
                if metric is None:
                    val = epoch + 1
                else:
                    val = metric[f'{item}']

                board.add_scalar(tag=tag, scalar_value=val, global_step=step)

        log_train.info(
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

            f'VA n={len(test_metric["truth"])} '
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
            log_train.info('Confusion Matrix')
            log_train.info(test_metric['cm'])

            log_test.info(f'[epoch {epoch + 1}/{n_epoch} | step {step}]')
            log_test.info('Train Confusion Matrix')
            log_test.info(train_metric['cm'])
            log_test.info(valid_metric['cm'])
            log_test.info(test_metric['cm'])
