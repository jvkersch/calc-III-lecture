import numpy as np
import matplotlib.pyplot as plt

def himmelblau(x, y):
    return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2

def himmelblau_gradient(x, y):
    return [
        4 * (x**2 + y - 11) * x + 2 * (x + y**2 - 7),
        2 * (x**2 + y - 11) + 4 * (x + y**2 - 7) * y,
    ]

x = y = np.linspace(-6, 6, 100)
X, Y = np.meshgrid(x, y)
Z = himmelblau(X, Y)
U, V = himmelblau_gradient(X, Y)

plt.streamplot(X, Y, -U, -V, color=np.log(Z))