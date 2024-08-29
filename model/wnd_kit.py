import os
import timeit
import logging

import torch
import numpy as np
from torch import nn
from torch.optim import Adam
import sklearn.metrics as metric
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

from model.tiny import TinySleepNet
from model.utsn import UTSN


class WndKit:
    def __init__(self, cfg, run_dir: str, fold: int, test=False):
        self.cfg = cfg
        self.log = logging.getLogger('train')

        self.fold = fold
        self.epoch = 0
        self.step = 0

        self.batch_size = self.cfg.train.batch_size
        self.loss_weight = self.cfg.model.loss_class_weight

        self.output_dir = os.path.join(run_dir, f'{fold:02d}')
        self.ckpt_dir = os.path.join(self.output_dir, 'ckpt')
        self.ckpt_best_dir = os.path.join(self.output_dir, 'ckpt_best')
        self.board_dir = os.path.join(self.output_dir, 'board')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model_name = self.cfg.model.name
        if self.model_name == 'utsn':
            self.model = UTSN(cfg)
        elif self.model_name == 'tiny':
            self.model = TinySleepNet(cfg)
        else:
            raise RuntimeError('No such named model')

        self.log.info(self.model)
        self.model.to(self.device)

        self.optim = Adam(
            self.model.parameters(),
            lr=self.cfg.train.lr,
            weight_decay=self.cfg.train.weight_decay
        )

        # self.sche = CosineAnnealingLR(self.optim, T_max=self.cfg.train.n_epoch)

        self.loss = nn.CrossEntropyLoss(
            reduction='none',
            weight=torch.Tensor(self.loss_weight).to(self.device)
        )

        self.tensorboard = SummaryWriter(
            log_dir=self.board_dir,
            comment=f'-train-{fold}' if not test else '-test'
        )

    def cur_epoch(self):
        return self.epoch

    def iter_epoch(self):
        self.epoch += 1

    def cur_step(self):
        return self.step

    def iter_step(self):
        self.step += 1

    def train_epoch(self, loader):
        t_start = timeit.default_timer()

        self.model.train()
        pred_s, truth_s, loss_s, metric_dict = ([], [], [], {})

        for (x, y, p) in loader:
            self.optim.zero_grad()

            x = x.to(self.device)
            y = y.long().to(self.device)
            p = p.to(self.device)

            if self.model_name == 'tiny':
                y_hat = self.model(x)
            elif self.model_name == 'utsn':
                y_hat = self.model(x, p)
            else:
                raise RuntimeError('No such named model')

            loss = self.loss(y_hat, y).sum()
            loss.backward()

            # 梯度裁剪
            # nn.utils.clip_grad_norm_(
            #     self.model.parameters(),
            #     max_norm=self.config['clip_grad_val'],
            #     norm_type=2
            # )
            self.optim.step()
            self.iter_step()

            loss_s.append(loss.detach().cpu().numpy())

            cur_pred = np.argmax(y_hat.detach().cpu().numpy(), axis=1).reshape(-1).tolist()
            cur_truth = y.detach().cpu().numpy().reshape(-1).tolist()

            pred_s.extend(cur_pred)
            truth_s.extend(cur_truth)

        mean_loss = np.array(loss_s).mean()
        acc = metric.accuracy_score(truth_s, pred_s)
        f1 = metric.f1_score(truth_s, pred_s, average='macro')
        cm = metric.confusion_matrix(truth_s, pred_s, labels=[0, 1, 2])

        t_stop = timeit.default_timer()
        dur = t_stop - t_start

        metric_dict.update({
            'global_step': self.cur_step(),
            'truth': truth_s,
            'pred': pred_s,
            'acc': acc,
            'loss': mean_loss,
            'f1': f1,
            'cm': cm,
            'dur': dur,
            # 'lr': self.sche.get_last_lr()[0]
            'lr': self.cfg.train.lr
        })

        # self.sche.step()
        self.iter_epoch()
        return metric_dict

    def eval_epoch(self, loader):
        t_start = timeit.default_timer()

        self.model.eval()

        pred_s, truth_s, loss_s, metric_dict = ([], [], [], {})

        with torch.no_grad():
            for (x, y, p) in loader:

                x = x.to(self.device)
                y = y.long().to(self.device)
                p = p.to(self.device)

                if self.model_name == 'tiny':
                    y_hat = self.model(x)
                elif self.model_name == 'utsn':
                    y_hat = self.model(x, p)
                else:
                    raise RuntimeError('No such named model')

                loss = self.loss(y_hat, y).sum()
                loss_s.append(loss.detach().cpu().numpy())

                cur_pred = np.argmax(y_hat.detach().cpu().numpy(), axis=1).reshape(-1).tolist()
                cur_truth = y.detach().cpu().numpy().reshape(-1).tolist()

                pred_s.extend(cur_pred)
                truth_s.extend(cur_truth)

        acc = metric.accuracy_score(truth_s, pred_s)
        f1 = metric.f1_score(truth_s, pred_s, average='macro')
        cm = metric.confusion_matrix(truth_s, pred_s, labels=[0, 1, 2])
        mean_loss = np.array(loss_s).mean()

        t_stop = timeit.default_timer()
        dur = t_stop - t_start

        metric_dict.update({
            'truth': truth_s,
            'pred': pred_s,
            'acc': acc,
            'loss': mean_loss,
            'f1': f1,
            'cm': cm,
            'dur': dur,
            # 'lr': self.sche.get_last_lr()[0]
            'lr': self.cfg.train.lr
        })

        return metric_dict

    def save_best_ckpt(self, name):
        if not os.path.exists(self.ckpt_best_dir):
            os.makedirs(self.ckpt_best_dir)
        save_path = os.path.join(self.ckpt_best_dir, f'{name}.ckpt')
        torch.save(self.model.state_dict(), save_path)
        self.log.info(f'Save best checkpoint at {save_path}')
