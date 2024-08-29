import os.path

import numpy as np
import pandas as pd

import scipy.signal as sig
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


# f, p = sig.periodogram(data, 1000.0, axis=2)
# f = f[1:lim + 1]
# p = p[:, :, 1:lim + 1]
# p = p.mean(axis=1)
# plt.semilogy(f, p[0], label='raw')

# f, p = sig.periodogram(data, 1000.0, axis=2)
# f = f[1:lim + 1]
# p = p[:, :, 1:lim + 1]
# p = p.mean(axis=1)
# plt.semilogy(f, p[0], label='lowpass')
#
# plt.legend()
# plt.show()
# plt.clf()

# a = data[0, 0, :]
# x = np.arange(0, 10.0, 1 / 1000.0)
# y = np.arange(0, 10.0, 1 / 100.0)
# cs = CubicSpline(x, data, axis=2)
# data = cs(y)

# plt.plot(x, a, label='1000')
# plt.plot(y, b, label='128')
#
# plt.legend()
# plt.show()
# plt.clf()


def filter_freq_visualize():
    sos = sig.butter(10, 45, 'lowpass', fs=1000, output='sos')
    w, h = sig.sosfreqz(sos, worN=10000, fs=1000)

    # plt.subplot(2, 1, 1)
    db = 20 * np.log10(np.maximum(np.abs(h), 1e-5))
    plt.plot(w[0:2000], db[0:2000])
    plt.grid(True)
    plt.ylabel('Gain [dB]')
    plt.title('Frequency Response')
    plt.show()
    plt.clf()


def lowpass_filter(data):
    sos = sig.butter(10, 45, 'lowpass', fs=1000, output='sos')
    data = sig.sosfiltfilt(sos, data, axis=2)
    return data


def load_from_file(idx: int):
    dataset_dir = '../dataset/mice/'
    idx_str = f'{idx:02d}'

    data_path = os.path.join(dataset_dir, idx_str, idx_str + '_wave.npy')
    label_path = os.path.join(dataset_dir, idx_str, idx_str + '_epoch.csv')

    data: np.ndarray = np.load(data_path)
    label: pd.DataFrame = pd.read_csv(label_path)

    return data, label


def split_sleep_type(label: np.ndarray):
    cls_arr = []
    for c in range(3):
        idx = np.where(label == c)[0]
        sel = np.zeros(shape=(idx.shape[0]), dtype=np.int32)
        for i, pos in enumerate(idx.tolist()):
            flag = True
            if pos - 1 >= 0:
                flag = flag & (label[pos-1] == c)
            if pos + 1 < label.shape[0]:
                flag = flag & (label[pos+1] == c)
            sel[i] = 1 if flag else 0
        sel = sel.astype(bool)
        cls_epoch = idx[sel]
        cls_arr.append(cls_epoch)
    return cls_arr


def data_filter(data: np.ndarray):
    d = np.zeros_like(data)
    eeg_sos = sig.butter(6, [0.3, 30], btype='bandpass', output='sos', fs=fs)
    emg_sos = sig.butter(6, [10, 100], btype='bandpass', output='sos', fs=fs)

    d[0:n_eeg_ch, :] = sig.sosfiltfilt(eeg_sos, data[0:n_eeg_ch, :], axis=1)
    d[n_eeg_ch:, :] = sig.sosfiltfilt(emg_sos, data[n_eeg_ch:, :], axis=1)

    return d


def calc_psd():
    freq = None
    psd_list = [[], [], []]
    for i in range(1, n_sub + 1):
        data, label = load_from_file(i)
        data = data_filter(data)

        data = data.reshape(n_ch, -1, wnd_pts)
        label = (label['class'].values - 1).reshape(-1)

        cls_idx = split_sleep_type(label)

        for j, c in enumerate(cls_name):
            idx = cls_idx[j]

            wave = data[:, idx, :]
            f, p = sig.periodogram(wave, fs, nfft=nfft, axis=2)
            f = f[0:lim + 0]
            p = p[:, :, 0:lim + 0]
            p = p.mean(axis=1)

            freq = f
            psd_list[j].append(p)

    psd = []
    for i in range(n_cls):
        psd.append(np.array(psd_list[i]))
    psd = np.array(psd)
    psd = np.transpose(psd, (2, 0, 1, 3))

    # shape: freq (lim,) , psd (n_ch, n_cls, n_sub, lim)
    return freq, psd


def calc_psd_norm(psd):
    psd_norm = np.zeros_like(psd)
    for i in range(n_ch):
        for j in range(n_cls):
            for k in range(n_sub):
                y = psd[i, j, k]

                total = y.sum()
                psd_norm[i, j, k] = (y / total) * 100
    # shape: psd (n_ch, n_cls, n_sub, lim)
    return psd_norm


def calc_psd_band_ratio(psd_norm: np.ndarray):
    shape = list(psd_norm.shape)
    shape[-1] = n_band

    psd_band = np.zeros(shape=shape)

    # [0.5, 2), [2, 4), [4, 10), [10, 20)
    band_pos = [2, 8, 16, 40, 80]

    for i in range(n_band):
        band_power = psd_norm[:, :, :, band_pos[i]:band_pos[i+1]].sum(axis=3)
        psd_band[:, :, :, i] = band_power

    # shape psd_band (n_ch, n_cls, n_sub, n_band)
    return psd_band


def draw_wave_with_var(x, y_mean, y_std, idx, name):
    palette = plt.get_cmap('Set1')
    color = palette(idx)

    down = (y_mean - y_std).tolist()
    up = (y_mean + y_std).tolist()

    plt.plot(x, y_mean, color=color, label=name, linewidth=1)
    plt.fill_between(x, down, up, color=color, alpha=0.2)


def draw_bar_with_var(x, y_mean, y_std, idx, name):
    width = 0.2
    bar_offset_arr = [-0.2, 0, 0.2]
    palette = plt.get_cmap('Set1')

    color = palette(idx)
    offset = bar_offset_arr[idx]
    x_offset = x + offset
    plt.bar(x_offset, y_mean, width,
            yerr=y_std, color=color, label=name,
            error_kw={'elinewidth': 1, 'ecolor': 'black', 'capsize': 3}
            )


n_cls = 3
cls_name = ['wake', 'nrem', 'rem']

n_ch = 4
n_eeg_ch = 2
n_sub = 10
lim = 100
nfft = 4000
fs = 1000.0
wnd_sec = 4.0
wnd_pts = round(fs * wnd_sec)

n_band = 4
x_band_pos = np.arange(n_band)
x_band_label = ['0.5-2', '2-4', '4-10', '10-20']


# shape: freq (lim,) , psd (n_ch, n_cls, n_sub, lim)
freq, psd = calc_psd()
psd_norm = calc_psd_norm(psd)
psd_band = calc_psd_band_ratio(psd_norm)

idx_sel = np.array([0, 2, 3, 4, 5, 7, 8])
psd_norm = psd_norm[:, :, idx_sel, :]
psd_band = psd_band[:, :, idx_sel, :]

# shape (n_ch, n_cls, lim)
psd_mean = np.mean(psd_norm, axis=2)
psd_std = np.std(psd_norm, axis=2)

# shape (n_ch, n_cls, n_band)
psd_band_mean = np.mean(psd_band, axis=2)
psd_band_std = np.std(psd_band, axis=2)


for i in range(n_ch):
    name_ch = f'M{i} {"EEG" if i < 2 else "EMG"}'

    plt.figure(figsize=(12, 6))
    plt.title(f'{name_ch} power spectrum')
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])

    for j, c in enumerate(cls_name):
        x = freq

        plt.subplot(1, 2, 1)
        draw_wave_with_var(freq, psd_mean[i, j], psd_std[i, j], j, c)

        plt.subplot(1, 2, 2)
        draw_bar_with_var(x_band_pos, psd_band_mean[i, j], psd_band_std[i, j], j, c)

    plt.subplot(1, 2, 1)
    plt.xlabel('Frequency (Hz)')
    # plt.ylabel('PSD [uV^2/Hz]')
    plt.ylabel('Normalized power (%)')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)

    plt.xticks(x_band_pos, x_band_label)
    plt.xlabel('Frequency Range (Hz)')
    plt.ylabel('Power Percentage Sum (a.u.)')
    plt.legend()

    plt.show()
    plt.clf()
