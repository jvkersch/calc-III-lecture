import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker


def himmelblau(x, y):
    return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2

def log_himmelblau_gradient(x, y):
    # himm = himmelblau(x, y)
    grad = np.asarray([
        4 * (x**2 + y - 11) * x + 2 * (x + y**2 - 7),
        2 * (x**2 + y - 11) + 4 * (x + y**2 - 7) * y,
    ])

    norm = np.linalg.norm(grad, axis=0) + 1e-3
    return grad / norm#[np.axis, :, :]


x = np.linspace(-6, 6, 100)
y = np.linspace(-6, 6, 100)
X, Y = np.meshgrid(x, y)
Z = himmelblau(X, Y)

Xs = X[::9, ::9]
Ys = Y[::9, ::9]
U, V = log_himmelblau_gradient(Xs, Ys)

plt.contourf(X, Y, Z, cmap="coolwarm", locator=ticker.LogLocator())
plt.quiver(Xs, Ys, -U, -V, angles='xy')
plt.title("Normalized negative gradient (Himmelblau function)")
plt.show()