import os

import numpy as np
import pandas as pd
import scipy.interpolate
from scipy import signal
from scipy.interpolate import CubicSpline

import matplotlib.pyplot as plt

# t = np.array([[1, 2], [3, 4]])
# t = np.power(t, 2)
# a = np.empty(shape=(0, *t.shape))
#
# df = pd.read_csv('../dataset/mice/extract/01/01_epoch.csv')
# label = (df['class'].values - 1).reshape(-1, 1)
# sub_start = np.arange(0, label.shape[0], 6)
# a = (label == 0).sum()
#
# label = label[sub_start: sub_start+6]

# print(df)

# data = np.load('../dataset/mice/extract/01/01_wave.npy')
# m1 = data[0, 0:1000]
# x = np.arange(0.0, 1.0, 0.001)
# y = np.arange(0.0, 1.0, 0.004)
#
# plt.plot(x, m1)
#
# cs = CubicSpline(x, m1)
# m1_new = cs(y)
#
# plt.plot(y, m1_new)
# plt.show()
# plt.clf()
#
# f, psd = signal.welch(m1, 1000.0)
#
#
# plt.semilogy(f, psd)
# plt.xlabel('Frequency [Hz]')
# plt.xlim([0.0001, 60])
# plt.ylabel('psd')
# plt.show()
#
# print(data.shape)

# a = np.array([0, 0, 1, 0, 1, 2, 1, 2])
# b = np.array_split(a, 4)
# c = a[b[2]]
# print(c)

# data = np.array([
#     [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
#     [[-1, -2, -3], [-4, -5, -6], [-7, -8, -9], [-10, -11, -12]]
# ])
# data = data.reshape(-1, 2, 2, 3)
# a = np.flip(data, 3)
# print(a)
# print(data)
# print(a.base == data)

from omegaconf import OmegaConf
conf = OmegaConf.load('../conf/utsn.yaml')
print(conf['loss_class_weight'])
