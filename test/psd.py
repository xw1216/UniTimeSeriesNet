import numpy as np
import pandas as pd

import scipy.signal as sig
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


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


data = np.load('../dataset/mice/01/01_wave.npy')
label = pd.read_csv('../dataset/mice/01/01_epoch.csv')
ch = data.shape[0]

fs = 1000.0

pt_wnd = 10000
data = data.reshape(ch, -1, pt_wnd)
label = (label['class'].values - 1).reshape(-1)

wake_idx = np.where(label == 0)[0]
nrem_idx = np.where(label == 1)[0]
rem_idx = np.where(label == 2)[0]


# Single savgol filter
# for (name, idx) in [
#     ('wake', wake_idx),
#     ('nrem', nrem_idx),
#     ('rem', rem_idx)
# ]:
#     wave = data[0, idx[0], :]
#     f, p = sig.periodogram(wave, 1000.0, axis=0)
#     # 光谱平滑滤波
#     p = sig.savgol_filter(p, 23, 2)
#
#     f = f[5:lim+5]
#     p = p[5:lim+5]
#
#     plt.semilogy(f, p, label=name)
#
# plt.title('M0 EEG Spectrum With Savgol Filter')
# plt.ylim((1, 1e4))
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('PSD [uV^2/Hz]')
# plt.grid(True)
# plt.legend()
# plt.show()
# plt.clf()

# a = data[0, 0, :]

# resample
x = np.arange(0, 10.0, 1 / 1000.0)
y = np.arange(0, 10.0, 1 / 100.0)
cs = CubicSpline(x, data, axis=2)
data = cs(y)

# plt.plot(x, a, label='1000')
# plt.plot(y, b, label='128')
#
# plt.legend()
# plt.show()
# plt.clf()

lim = 100
info_list = []
for (name, idx) in [
    ('wake', wake_idx),
    ('nrem', nrem_idx),
    ('rem', rem_idx)
]:

    wave = data[:, idx, :]
    f, p = sig.periodogram(wave, 1000.0, axis=2, nfft=4000)
    f = f[1:lim+1]
    p = p[:, :, 5:lim+5]
    p = p.mean(axis=1)
    info_list.append((f, p))

for i in range(4):
    label = f'M{i} {"EEG" if i < 2 else "EMG"}'

    plt.figure(figsize=(10, 6))
    plt.title(f'{label} power spectral')
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])

    width = 0.2
    bar_offset_arr = [-0.2, 0, 0.2]
    x_band = np.arange(4)
    x_band_label = ['0.5-2', '2-4', '4-10', '10-20']

    for j, c in enumerate(['wake', 'nrem', 'rem']):
        x = info_list[j][0]
        y = info_list[j][1][i]

        total = y.sum()
        y_norm = (y / total) * 100

        plt.subplot(1, 2, 1)
        plt.plot(x, y_norm, label=c)

        low_delta = y_norm[1:7].sum()    # [0.5, 2)
        high_delta = y_norm[7:16].sum()  # [2, 4)
        theta = y_norm[16: 40].sum()     # [4, 10)
        alpha = y_norm[40: 80].sum()     # [10, 20)
        y_band = np.array([low_delta, high_delta, theta, alpha])

        plt.subplot(1, 2, 2)
        plt.bar(x_band + bar_offset_arr[j], y_band, width, label=c)

        # plt.semilogy(x, y, label=c)

    plt.subplot(1, 2, 1)

    # plt.ylim((1, 1e4))
    plt.xlabel('Frequency (Hz)')
    # plt.ylabel('PSD [uV^2/Hz]')
    plt.ylabel('Normalized power (%)')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)

    plt.xticks(x_band, x_band_label)
    plt.xlabel('Frequency Range (Hz)')
    plt.ylabel('Percentage Sum (a.u.)')
    # plt.grid(True)
    plt.legend()

    plt.show()
    plt.clf()