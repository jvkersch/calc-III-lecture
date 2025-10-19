import numpy as np
import plotly.graph_objects as go


def himmelblau(x, y):
    return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2


def himmelblau_gradient(x, y):
    return np.asarray(
        [
            4 * (x**2 + y - 11) * x + 2 * (x + y**2 - 7),
            2 * (x**2 + y - 11) + 4 * (x + y**2 - 7) * y,
        ]
    )


def himmelblau_hessian(x, y):
    return np.asarray(
        [
            [4 * (x**2 + y - 11) + 8 * x**2 + 2, 4 * x + 4 * y],
            [4 * x + 4 * y, 4 * (x + y**2 - 7) + 8 * y**2 + 2],
        ]
    )


def newton_one_step(x, y):
    update = np.linalg.solve(himmelblau_hessian(x, y), himmelblau_gradient(x, y))
    return x - update[0], y - update[1]


def gradient_descent_one_step(x, y, h=0.01):
    update = himmelblau_gradient(x, y)
    return x - h * update[0], y - h * update[1]


def run_n(method, start, n=10):
    return np.asarray([start] + [(start := method(*start)) for i in range(n)])


start = (4.5, 3)
newton = run_n(newton_one_step, start, n=10)
gradient = run_n(gradient_descent_one_step, start, n=10)

x = np.linspace(1, 5, 100)
y = np.linspace(0, 4, 100)
X, Y = np.meshgrid(x, y)
Z = himmelblau(X, Y)

fig = go.Figure()

# Filled contours (heatmap style) with a coolwarm-like palette (RdBu reversed)
fig.add_trace(
    go.Contour(
        x=x,  # meshgrid X axis
        y=y,  # meshgrid Y axis
        z=np.log10(Z),  # log scale similar to LogLocator
        showscale=False,
        contours=dict(coloring="heatmap", showlines=False),
        colorscale="RdBu",
        reversescale=True,  # RdBu reversed ≈ Matplotlib coolwarm
        hovertemplate="x=%{x}<br>y=%{y}<br>log10(Z)=%{z}<extra></extra>",
        name="Contours",
    )
)

# Gradient descent path (black, with markers)
fig.add_trace(
    go.Scatter(
        x=gradient[:, 0],
        y=gradient[:, 1],
        mode="lines+markers",
        line=dict(width=2, color="black"),
        marker=dict(size=6, color="black"),
        name="Gradient",
    )
)

# Newton path (red, with markers)
fig.add_trace(
    go.Scatter(
        x=newton[:, 0],
        y=newton[:, 1],
        mode="lines+markers",
        line=dict(width=2, color="red"),
        marker=dict(size=6, color="red"),
        name="Newton",
    )
)

fig.update_layout(
    xaxis_title="X",
    yaxis_title="Y",
    legend_title_text="",
)

fig.show()
