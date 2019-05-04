import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

d = 2
num = 3000
max_iter = 50
save_animation = True

x = np.linspace(-2, 2, 2*num)
y = np.linspace(-2, 0, num).reshape(-1, 1)
c = x + y * 1j
z = np.zeros(c.shape) + 0j
q = (x - 0.25) ** 2 + y * y
check = q * (q + x - 0.25) > 0.25 * y * y
in_set = np.ones(c.shape, dtype=bool)
iterations = np.zeros(c.shape, dtype=int)

if d == 2:
    in_set &= check

fig = plt.figure()
plt.axis('off')
img = plt.imshow(np.vstack((iterations, np.flipud(iterations))), cmap='gnuplot2')


def init():
    global z, in_set, iterations
    z = np.zeros(c.shape) + 0j
    in_set = np.ones(c.shape, dtype=bool)
    iterations = np.zeros(c.shape, dtype=int)
    img.set_data(np.vstack((iterations, np.flipud(iterations))))

    if d == 2:
        in_set &= check

    return img


def update(i):
    global z, in_set, iterations
    z[in_set] = z[in_set] ** d + c[in_set]
    zr = z[in_set].real
    zi = z[in_set].imag
    in_set_prev = np.copy(in_set)
    in_set[in_set] = zr * zr + zi * zi <= 4
    iterations[~in_set & in_set_prev] = i+1

    img.set_data(np.vstack((iterations, np.flipud(iterations))))
    img.set_clim(0, i)

    return img


animation = FuncAnimation(fig, update, init_func=init, frames=max_iter)

if save_animation:
    animation.save('./assets/mandelbrot-d{}.gif'.format(d), writer='imagemagick', fps=5)
else:
    plt.show()
