import os
import numpy as np
import matplotlib.pyplot as plt
from hampel import hampel
from scipy.signal import medfilt
import scipy.signal as sig

dataset_dir = '..//dataset//mice'

sos = sig.butter(2, [2, 45], btype='bandpass', fs=1000.0, output='sos')

ch = 2
pts = 86400000
x = np.arange(pts)

for i in range(1, 10 + 1):
    file_path = os.path.join(dataset_dir, f'{i:02d}', f'{i:02d}_wave.npy')
    data = np.load(file_path)

    data: np.ndarray = data[ch]
    data = sig.sosfiltfilt(sos, data)

    mean = data.mean()
    std = data.std()

    plt.title(f'Sub {i:02d} Ch {ch} Waveform')
    plt.plot(x, data)

    plt.show()
    plt.clf()

    max_idx = (data > 600)
    min_idx = (data < -850)
    data[max_idx] = mean
    data[min_idx] = mean

    plt.title(f'Sub {i:02d} Ch {ch} Waveform Median Filter')
    plt.plot(x, data)

    plt.show()
    plt.clf()


