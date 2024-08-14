import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline

data = np.load('../dataset/mice/extract/01/01_wave.npy')
m0 = data[0, 0:1000]
x = np.arange(0.0, 1.0, 1 / 1000)
y1 = np.arange(0.0, 1.0, 1 / 250)
y2 = np.arange(0.0, 1.0, 1 / 128)

plt.plot(x, m0, label='1000Hz')

cs = CubicSpline(x, m0)
# m1 = cs(y1)
m2 = cs(y2)

# plt.plot(y1, m1, label='250Hz')
plt.plot(y2, m2, label='128Hz')
plt.grid()
plt.legend()
plt.show()
plt.clf()
