import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

num = 3000
max_iter = 50

x = np.linspace(-2, 1, 2*num)
y = np.linspace(-1, 0, num).reshape(-1, 1)
c = x + y * 1j
z = np.zeros(c.shape) + 0j
in_set = np.ones(c.shape, dtype=bool)
iterations = np.zeros(c.shape, dtype=int)

fig = plt.figure()
plt.axis('off')
img = plt.imshow(np.vstack((iterations, np.flipud(iterations))), cmap='gnuplot2')


def init():
    global z, in_set, iterations
    z = np.zeros(c.shape) + 0j
    in_set = np.ones(c.shape, dtype=bool)
    iterations = np.zeros(c.shape, dtype=int)
    img.set_data(np.vstack((iterations, np.flipud(iterations))))

    return img


def update(i):
    global z, in_set, iterations
    z[in_set] = z[in_set] * z[in_set] + c[in_set]
    in_set = z.real*z.real + z.imag*z.imag <= 4
    iterations[in_set] += 1

    img.set_data(np.vstack((iterations, np.flipud(iterations))))
    img.set_clim(0, i)

    return img


animation = FuncAnimation(fig, update, init_func=init, frames=max_iter)
plt.show()
