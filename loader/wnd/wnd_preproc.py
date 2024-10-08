import logging
import os.path
import timeit

import numpy as np
import pandas as pd

from numpy import ndarray
from scipy.interpolate import CubicSpline
from scipy.signal import periodogram

import matplotlib.pyplot as plt
import scipy.signal as sig


class WndPreproc:
    def __init__(self, cfg):
        self.cfg = cfg
        self.log = logging.getLogger('train')
        self.data_dir: str = cfg.dataset.path
        self.chs = np.array(cfg.dataset.chs)

        self.fs_orig = round(cfg.dataset.fs_orig)
        self.fs = round(cfg.dataset.fs)
        self.filt_range = cfg.dataset.filt_range
        self.amp_range = cfg.dataset.amp_range

        self.wnd = round(cfg.dataset.wnd_len)
        self.pts = round(self.wnd * self.fs)

        self.sub_idx = []
        self.scan_idx()

    def scan_idx(self):
        for idx in os.listdir(self.data_dir):
            self.sub_idx.append(idx)

        # select correct alpha band mice subject
        if self.cfg.dataset.name == 'mice':
            self.sub_idx = ['01', '03', '04', '05', '06', '08', '09']

    def load_mice_raw_file(self, idx):
        idx_str: str = self.sub_idx[idx-1]
        path = os.path.join(self.data_dir, idx_str)
        data = np.load(os.path.join(path, idx_str + '_wave.npy'))
        data: ndarray = data[self.chs, :].reshape(self.chs.shape[0], -1)

        label: pd.DataFrame = pd.read_csv(os.path.join(path, idx_str + '_epoch.csv'))
        label: ndarray = (label['class'].values - 1).reshape(1, -1)

        return data, label

    def load_sleep_edf_raw_file(self, idx):
        prefix = 'SC4'
        subfix = 'E0.npz'

        path = os.path.join(self.data_dir, self.sub_idx[idx-1])
        data, label = [], []
        for i in range(2):
            file_name = prefix + self.sub_idx[idx-1] + str(i+1) + subfix
            file_path = os.path.join(path, file_name)
            npz = np.load(file_path)

            x, y = npz['x'], npz['y']

            y[(y > 0) & (y < 4)] = 1
            y[y == 4] = 2

            data.append(x)
            label.append(y)

        data = np.concatenate(data, axis=0)
        label = np.concatenate(label, axis=0)
        data = np.transpose(data, (2, 0, 1))
        label = label.reshape(1, -1)

        return data, label

    def filter(self, data: ndarray):
        if self.cfg.dataset.use_filter:
            sos = sig.butter(2, self.filt_range, btype='bandpass', fs=self.fs_orig, output='sos')
            data = sig.sosfiltfilt(sos, data, axis=1)

        return data

    def amp_limit(self, data, label):
        if self.cfg.dataset.use_amp_limit:
            amp_sel = (data < self.amp_range[0]) | (data > self.amp_range[1])
            amp_sel = (np.sum(amp_sel, axis=2) == 0)

            sel_arr = np.arange(data.shape[1])
            for i in range(self.chs.shape[0]):
                sel_ch = amp_sel[i]
                sel_ch = np.where(sel_ch > 0)[0]
                sel_arr = np.intersect1d(sel_arr, sel_ch)

            data = data[:, sel_arr, :]
            label = label[:, sel_arr]

        return data, label

    def resample(self, data: ndarray):
        pts = data.shape[1]
        if abs(self.fs - self.fs_orig) < 1.0:
            return data.reshape(self.chs.shape[0], -1, self.pts)

        t_orig = np.arange(pts)
        t = np.arange(round(pts * self.fs / self.fs_orig))

        cs = CubicSpline(t_orig, data, axis=1)
        data = cs(t).reshape(self.chs.shape[0], -1, self.pts)
        return data

    def aug_noise(self, signal: ndarray):
        if self.cfg.augment.noise:
            snr = np.random.choice(self.cfg.augment.noise_snr)
            noise = np.random.randn(*signal.shape)

            signal_power = (1 / signal.shape[1]) * np.sum(np.power(signal, 2), axis=2, keepdims=True)
            k = signal_power / np.power(10, (snr / 10))
            noise = np.sqrt(k) * noise
            signal += noise

        return signal

    def aug_scale(self, data: ndarray):
        if self.cfg.augment.scale:
            lim = self.cfg.augment.scale_lim
            arr = np.arange(*self.cfg.augment.scale_range, self.cfg.augment.scale_step)
            arr = np.delete(arr, ((1 - lim <= arr) & (arr <= 1 + lim)))
            rate = np.random.choice(arr, size=(data.shape[0], data.shape[1], 1))

            return data * rate
        else:
            return data

    def aug_roll(self, data: ndarray):
        if self.cfg.augment.roll:
            rate = self.cfg.augment.roll_rate
            lim = self.cfg.augment.roll_lim
            offset = round(self.pts * rate)

            arr = np.arange(-offset, offset + 1)
            arr = np.delete(arr, (-lim < arr) & (arr < lim))
            shift = np.random.choice(arr, size=(data.shape[0], data.shape[1]))

            n_chs = data.shape[0]
            for i in range(n_chs):
                data[i] = np.roll(data[i], shift=shift[i], axis=1)

        return data

    def calc_spectrum(self, data):
        bin_n = self.cfg.model.n_spec_freq
        if not self.cfg.model.use_spec:
            data_shape = list(data.shape)
            data_shape[2] = bin_n
            f = np.zeros(shape=(bin_n,))
            p = np.zeros(shape=data_shape)
            return f, p

        bin_len = self.cfg.model.bin_len
        freq_range = self.cfg.model.fft_freq_range
        start_idx = round(freq_range[0] / bin_len)

        f, p = periodogram(data, self.fs, axis=2, nfft=round(self.fs * (1 / bin_len)))

        f = f[start_idx:start_idx + bin_n]
        p = p[:, :, start_idx:start_idx + bin_n]
        return f, p

    def aug(self, data, label):
        is_aug_class = self.cfg.augment.aug_class
        aug_ratio = self.cfg.augment.aug_ratio
        data_gen = np.empty((data.shape[0], 0, data.shape[2]))
        label_gen = np.empty((1, 0))

        for i in range(len(is_aug_class)):
            cls_idx = np.squeeze((label == i))
            x = data[:, cls_idx, :]
            y = label[:, cls_idx]

            data_gen = np.concatenate((data_gen, x), axis=1)
            label_gen = np.concatenate((label_gen, y), axis=1)

            if is_aug_class[i]:
                for j in range(aug_ratio[i]):
                    x_aug = self.aug_noise(x)
                    x_aug = self.aug_scale(x_aug)
                    x_aug = self.aug_roll(x_aug)

                    data_gen = np.concatenate((data_gen, x_aug), axis=1)
                    label_gen = np.concatenate((label_gen, y), axis=1)

        shuffle = self.cfg.augment.shuffle
        if shuffle:
            shuffle_idx = np.arange(label_gen.shape[1])
            np.random.shuffle(shuffle_idx)

            data_gen = data_gen[:, shuffle_idx, :]
            label_gen = label_gen[:, shuffle_idx]

        return data_gen, label_gen

    def preproc_sub(self, idx, is_test):
        self.log.info(f'Sub {self.sub_idx[idx-1]} processing')
        t_start = timeit.default_timer()
        if self.cfg.dataset.name == 'mice':
            x, y = self.load_mice_raw_file(idx)
        elif self.cfg.dataset.name == 'sleepedf':
            x, y = self.load_sleep_edf_raw_file(idx)
        else:
            raise RuntimeError('Do such named dataset')

        x = self.filter(x)
        x = self.resample(x)
        x, y = self.amp_limit(x, y)

        if not is_test:
            x, y = self.aug(x, y)

        _, p = self.calc_spectrum(x)

        if self.cfg.dataset.use_max_min_norm:
            x = self.max_min_norm(x)
            p = self.max_min_norm(p)

        t_end = timeit.default_timer()
        t_span = t_end - t_start
        self.log.info(f'Sub {self.sub_idx[idx-1]} done, t={t_span:.2f}')
        return x, y, p

    def max_min_norm(self, data: ndarray):
        if not self.cfg.model.use_spec:
            return data
        max_val = np.max(data, axis=2, keepdims=True)
        min_val = np.min(data, axis=2, keepdims=True)
        data = (data - min_val) / (max_val - min_val)
        return data

    def split_set(self, fold):
        n_sub = self.cfg.dataset.n_sub
        n_fold = self.cfg.train.n_fold

        idx_arr = np.arange(1, n_sub + 1)
        idx_split = np.array_split(idx_arr, n_fold)

        test_idx = idx_split[fold]
        train_idx = np.setdiff1d(idx_arr, test_idx)
        valid_idx = np.random.choice(train_idx, size=(1,))
        train_idx = np.setdiff1d(train_idx, valid_idx)
        return train_idx, valid_idx, test_idx

    @staticmethod
    def set_merge(dataset):
        if len(dataset) < 1:
            raise RuntimeError('Empty Dataset')
        x_shape = list(dataset[0][0].shape)
        p_shape = list(dataset[0][2].shape)
        x_shape[1] = 0
        p_shape[1] = 0
        y_shape = [1, 0]

        x, y, p = np.empty(x_shape), np.empty(y_shape), np.empty(p_shape)
        for i in range(len(dataset)):
            x = np.concatenate((x, dataset[i][0]), axis=1)
            y = np.concatenate((y, dataset[i][1]), axis=1)
            p = np.concatenate((p, dataset[i][2]), axis=1)

        x = np.transpose(x, (1, 0, 2))
        y = np.squeeze(y)
        p = np.transpose(p, (1, 0, 2))
        return x, y, p

    def preproc_fold(self, fold):
        train_idx, valid_idx, test_idx = self.split_set(fold)

        self.log.info(f'Train Set {train_idx}')
        self.log.info(f'Valid Set {valid_idx}')
        self.log.info(f'Test Set {test_idx}')

        train_dataset = []
        for i in train_idx:
            train_dataset.append(self.preproc_sub(i, False))

        valid_dataset = []
        for i in valid_idx:
            valid_dataset.append(self.preproc_sub(i, True))

        test_dataset = []
        for i in test_idx:
            test_dataset.append(self.preproc_sub(i, True))

        train_dataset = self.set_merge(train_dataset)
        valid_dataset = self.set_merge(valid_dataset)
        test_dataset = self.set_merge(test_dataset)

        return train_dataset, valid_dataset, test_dataset
