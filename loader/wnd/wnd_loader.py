import enum

import numpy as np
import torch
from numpy import ndarray
from torch.utils.data import Dataset


class WndDataset(Dataset):
    def __init__(self, data: tuple[ndarray, ndarray, ndarray]):
        self.x = data[0]
        self.y = data[1]
        self.p = data[2]

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx], self.p[idx]

    def get(self, idx: ndarray):
        return self.x[idx], self.y[idx], self.p[idx]


class WndLoader:
    def __init__(self, cfg, dataset):
        self.cfg = cfg
        self.dataset = WndDataset(dataset)
        self.batch_size = cfg.train.batch_size
        self.batch_size_test = cfg.train.batch_size_test
        n_sample = len(self.dataset)
        self.n_batch = ((n_sample // self.batch_size) +
                        (1 if n_sample % self.batch_size > 0 else 0))
        sample_idx = np.arange(n_sample)
        np.random.shuffle(sample_idx)

        self.batch_arr = np.array_split(sample_idx, self.n_batch)

    def iter(self):
        for b in range(self.n_batch):
            b_idx = self.batch_arr[b]
            (data, label, power) = self.dataset.get(b_idx)
            if data.ndim < 3:
                data = np.expand_dims(data, 0)
                power = np.expand_dims(power, 0)
                label = label.reshape(1,)

            yield torch.Tensor(data), torch.Tensor(label), torch.Tensor(power)

