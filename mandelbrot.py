import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

num = 3000
max_iter = 50

x = np.linspace(-2, 1, 2*num)
y = np.linspace(-1, 0, num).reshape(-1, 1)
c = x + y * 1j
z = np.zeros(c.shape) + 0j
q = (x - 0.25) ** 2 + y * y
check = q * (q + x - 0.25) > 0.25 * y * y
in_set = np.ones(c.shape, dtype=bool) & check
iterations = np.zeros(c.shape, dtype=int)

fig = plt.figure()
plt.axis('off')
img = plt.imshow(np.vstack((iterations, np.flipud(iterations))), cmap='gnuplot2')


def init():
    global z, in_set, iterations
    z = np.zeros(c.shape) + 0j
    in_set = np.ones(c.shape, dtype=bool) & check
    iterations = np.zeros(c.shape, dtype=int)
    img.set_data(np.vstack((iterations, np.flipud(iterations))))

    return img


def update(i):
    global z, in_set, iterations
    z[in_set] = z[in_set] * z[in_set] + c[in_set]
    zr = z[in_set].real
    zi = z[in_set].imag
    in_set_prev = np.copy(in_set)
    in_set[in_set] = zr * zr + zi * zi <= 4
    iterations[~in_set & in_set_prev] = i+1

    img.set_data(np.vstack((iterations, np.flipud(iterations))))
    img.set_clim(0, i)

    return img


animation = FuncAnimation(fig, update, init_func=init, frames=max_iter)
plt.show()
