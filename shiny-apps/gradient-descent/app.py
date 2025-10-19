from shiny import reactive
from shiny.express import input, render, ui
from shiny.ui import output_plot
from shinywidgets import render_plotly

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


### Objective functions ###


class ObjFunQuadratic:
    X_BOUNDS = [-3.0, 3.0]
    Y_BOUNDS = [-3.0, 3.0]

    def eval_fun(self, x, y):
        return x**2 + y**2 / 3

    def eval_grad(self, x, y):
        return [2 * x, 2 * y / 3]


class ObjFunHimmelblau:
    X_BOUNDS = [-4.0, 4.0]
    Y_BOUNDS = [-4.0, 4.0]

    def eval_fun(self, x, y):
        return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2

    def eval_grad(self, x, y):
        return [
            4 * (x**2 + y - 11) * x + 2 * (x + y**2 - 7),
            2 * (x**2 + y - 11) + 4 * (x + y**2 - 7) * y,
        ]


class ObjFunLogRosenbrock:
    X_BOUNDS = [-3.0, 3.0]
    Y_BOUNDS = [-3.0, 3.0]

    def eval_fun(self, x, y):
        return np.log((1 - x) ** 2 + 100 * (y - x**2) ** 2 + 1)

    def eval_grad(self, x, y):
        f = np.exp(self.eval_fun(x, y))
        g = [
            (-2 * (1 - x) - 400 * x * (y - x**2)) / (f + 1),
            (200 * (y - x**2)) / (f + 1),
        ]
        return g


FUNCTIONS = {
    "Quadratic": ObjFunQuadratic(),
    "Himmelblau": ObjFunHimmelblau(),
    "log-Rosenbrock": ObjFunLogRosenbrock(),
}

### Running list of trajectories that have been explored ###

# Clicking the step button adds a new point to the last (active) trajectory. CLicking
# anywhere on the plot starts a new trajectory. The _TRAJECTORIES reactive is semi-
# private, since it needs to be set with a copy to trigger updates in the UI.

_TRAJECTORIES = reactive.Value([])


def _update_trajectories(pts):
    _TRAJECTORIES.set(pts.copy())  # set with copy to trigger update


def start_new_trajectory(pt):
    pts = _TRAJECTORIES()
    pts.append([pt])
    _update_trajectories(pts)


def add_to_trajectory(pt):
    pts = _TRAJECTORIES()
    pts[-1].append(pt)
    _update_trajectories(pts)


def reset_trajectories():
    _TRAJECTORIES.set([])


def get_trajectories():
    return _TRAJECTORIES()


### UI code ###

ui.page_opts(title="Gradient descent: trajectories", fillable=True)

def get_obj():
    return FUNCTIONS[input.function()]

with ui.sidebar():
    ui.input_selectize(
        "function",
        "Function",
        list(FUNCTIONS),
    )
    ui.input_slider("stepsize", "Step size", 0.01, 1.0, 0.1)
    ui.input_checkbox("adaptive", "Adaptive step size")
    ui.input_checkbox("vector_field", "Show negative gradient")
    ui.input_action_button("step_btn", "Step", class_="btn-success", disabled=True)
    ui.input_action_button("reset_btn", "Reset")


with ui.layout_columns(col_widths=(12, )):
    with ui.card():
        output_plot(
            "image_2d",
            click=True,
        )

        with ui.hold():

            @render.plot
            def image_2d():
                obj = get_obj()

                x = np.linspace(*obj.X_BOUNDS)
                y = np.linspace(*obj.Y_BOUNDS)
                X, Y = np.meshgrid(x, y)

                # plot objective function
                Z = obj.eval_fun(X, Y)
                plt.contourf(X, Y, Z, cmap="coolwarm")
                plt.colorbar()
                plt.axis("equal")

                h = input.stepsize()

                # plot gradient
                if input.vector_field():
                    xg = np.linspace(*obj.X_BOUNDS, 10)
                    yg = np.linspace(*obj.Y_BOUNDS, 10)
                    XG, YG = np.meshgrid(xg, yg)
                    U, V = obj.eval_grad(XG, YG)

                    plt.quiver(XG, YG, -U, -V)

                # plot trajectories on top of contour map
                for traj in get_trajectories():
                    traj = np.asarray(traj)
                    plt.plot(traj[:, 0], traj[:, 1], "r.")

                pts = get_trajectories()
                if len(pts) > 0:
                    current = pts[-1][-1]
                    u, v = obj.eval_grad(*current)
                    plt.quiver(
                        *current,
                        -h * u,
                        -h * v,
                        color="red",
                        angles="xy",
                        scale_units="xy",
                        scale=1,
                    )

                plt.tight_layout()

    # with ui.card():

    #     @render_plotly
    #     def image_3d():
    #         obj = get_obj()

    #         x = np.linspace(*obj.X_BOUNDS)
    #         y = np.linspace(*obj.Y_BOUNDS)
    #         X, Y = np.meshgrid(x, y)

    #         Z = obj.eval_fun(X, Y)

    #         fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z)])
    #         fig.update_layout(
    #             autosize=False,
    #             width=500,
    #             height=500,
    #             # margin=dict(l=65, r=50, b=65, t=90),
    #         )
    #         return fig

    @reactive.effect
    @reactive.event(input.image_2d_click)
    def update_starting_point():
        x = input.image_2d_click()["x"]
        y = input.image_2d_click()["y"]
        start_new_trajectory((x, y))

        ui.update_action_button("step_btn", disabled=False)

    @reactive.effect
    @reactive.event(input.reset_btn, input.adaptive)
    def reset_btn_clicked():
        reset_trajectories()

        ui.update_action_button("step_btn", disabled=True)

    @reactive.effect
    @reactive.event(input.step_btn)
    def step_btn_clicked():
        pts = get_trajectories()
        obj = get_obj()

        # Invariant: step button is disabled when there is no starting point,
        # so the `pts` array is never empty
        current = pts[-1][-1]

        if input.adaptive():
            nxt = next_step_gradient_adaptive(obj, current, 0.1)
        else:
            nxt = next_step_gradient_descent(obj, current, input.stepsize())

        add_to_trajectory(nxt)


def next_step_gradient_descent(obj, start, h0):
    grad = obj.eval_grad(*start)
    return [start[0] - h0 * grad[0], start[1] - h0 * grad[1]]


def next_step_gradient_adaptive(obj, start, h0):
    loss_current = obj.eval_fun(*start)
    for i in range(10):
        nxt = next_step_gradient_descent(obj, start, h0)
        loss_nxt = obj.eval_fun(*nxt)
        if loss_nxt < loss_current:
            h0 *= 1.10  # slightly larger timestep
        else:
            h0 *= 0.70  # overshoot: shrink timestep
            nxt = start
            loss_nxt = loss_current

    return nxt
