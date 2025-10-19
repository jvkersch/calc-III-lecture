from shiny import reactive
from shiny.express import input, render, ui
from shinywidgets import render_plotly


import numpy as np
import plotly.graph_objects as go


### Objective functions ###


class ObjFunQuadratic:
    X_BOUNDS = [-3.0, 3.0]
    Y_BOUNDS = [-3.0, 3.0]

    GRAD_TOL = 2.0

    def eval_fun(self, x, y):
        return x**2 + y**2 / 3

    def eval_grad(self, x, y):
        return np.asarray([2 * x, 2 * y / 3])

    def eval_hessian(self, x, y):
        return np.asarray([[2, 0], [0, 2 / 3]])


class ObjFunTwoMinima:
    X_BOUNDS = [-2.0, 2.0]
    Y_BOUNDS = [-1.0, 1.0]

    GRAD_TOL = 2.0

    def eval_fun(self, x, y):
        return x**4 - 4 * x**2 + y**2

    def eval_grad(self, x, y):
        return np.asarray(
            [
                4 * x**3 - 8 * x,
                2 * y,
            ]
        )

    def eval_hessian(self, x, y):
        return np.asarray([[12 * x**2 - 8, 0], [0, 2]])


class ObjFunHimmelblau:
    X_BOUNDS = [-4.0, 4.0]
    Y_BOUNDS = [-4.0, 4.0]

    GRAD_TOL = 15.0

    def eval_fun(self, x, y):
        return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2

    def eval_grad(self, x, y):
        return np.asarray(
            [
                4 * (x**2 + y - 11) * x + 2 * (x + y**2 - 7),
                2 * (x**2 + y - 11) + 4 * (x + y**2 - 7) * y,
            ]
        )

    def eval_hessian(self, x, y):
        return np.asarray(
            [
                [4 * (x**2 + y - 11) + 8 * x**2 + 2, 4 * x + 4 * y],
                [4 * x + 4 * y, 4 * (x + y**2 - 7) + 8 * y**2 + 2],
            ]
        )


FUNCTIONS = {
    "Quadratic": ObjFunQuadratic(),
    "Two minima": ObjFunTwoMinima(),
    "Himmelblau": ObjFunHimmelblau(),
}

### Placeholder for mouse hover over coordinates ###

hover_reactive = reactive.value()


def on_point_hover(_, points, __):
    coords = points.xs[0], points.ys[0]
    hover_reactive.set(coords)


### UI ###

ui.page_opts(title="The second derivative test", fillable=True)

ui.head_content(
    ui.tags.script(src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js")
)


def _safe_round(x):
    return "-" if x is None else round(x, 2)


def _make_gradient(dx=None, dy=None):
    return rf"""Gradient:
\begin{{bmatrix}}
    {_safe_round(dx)} & {_safe_round(dy)}
\end{{bmatrix}}
"""


def _make_hessian(dxx=None, dxy=None, dyy=None):
    return rf"""Hessian:
\begin{{bmatrix}}
    {_safe_round(dxx)} & {_safe_round(dxy)} \\\\
    {_safe_round(dxy)} & {_safe_round(dyy)}
\end{{bmatrix}}
"""


def _wrap_mathjax(latex):
    return ui.TagList(
        ui.HTML(latex), ui.tags.script("window.MathJax && MathJax.typeset();")
    )


def get_obj():
    return FUNCTIONS[input.function()]


with ui.sidebar():
    ui.input_selectize(
        "function",
        "Function",
        list(FUNCTIONS),
    )

    @render.ui
    def show_gradient():
        obj = get_obj()
        coords = hover_reactive.get()
        u, v = obj.eval_grad(*coords)
        latex = _make_gradient(u, v)
        return _wrap_mathjax(latex)

    @render.ui
    def show_hessian():
        obj = get_obj()
        coords = hover_reactive.get()
        h = obj.eval_hessian(*coords)
        latex = _make_hessian(h[0, 0], h[0, 1], h[1, 1])
        return _wrap_mathjax(latex)

    @render.ui
    def show_classification():
        obj = get_obj()
        coords = hover_reactive.get()
        g = obj.eval_grad(*coords)
        h = obj.eval_hessian(*coords)
        d = np.linalg.det(h)

        if np.linalg.norm(g) > obj.GRAD_TOL:
            cls = "-"
        else:
            if d > 0:
                cls = "minimum" if h[0, 0] > 0 else "maximum"
            elif d < 0:
                cls = "saddle"

        return f"Classification: {cls}"


with ui.card():

    @render_plotly
    def image_3d():
        obj = get_obj()

        x = np.linspace(*obj.X_BOUNDS)
        y = np.linspace(*obj.Y_BOUNDS)
        X, Y = np.meshgrid(x, y)
        Z = obj.eval_fun(X, Y)

        fig = go.Figure(
            data=[
                go.Surface(
                    x=X,
                    y=Y,
                    z=Z,
                    hoverinfo="none",
                )
            ]
        )
        fig.update_layout(
            autosize=False,
            width=500,
            height=500,
        )

        w = go.FigureWidget(fig)
        w.data[0].on_hover(on_point_hover)

        return w
