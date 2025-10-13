from shiny import reactive
from shiny.express import input, render, ui

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from skimage import data
from skimage.color import rgb2gray
from skimage.filters import gaussian

ui.page_opts(fillable=True)

GRAD_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

GRAD_X = GRAD_Y.T


def cat_factory():
    return rgb2gray(data.cat())


def square_factory():
    square = np.zeros((400, 400), dtype=float)
    square[100:300, 100:300] = 1.0
    return square


def coins_factory():
    return data.coins()


def black_white_horizontal_factory():
    arena = np.zeros((400, 400), dtype=float)
    arena[200:, :] = 1.0
    return gaussian(arena, sigma=10.0)


def black_white_vertical_factory():
    arena = np.zeros((400, 400), dtype=float)
    arena[:, 200:] = 1.0
    return gaussian(arena, sigma=10.0)


IMAGE_FACTORIES = {
    "Square": square_factory,
    "Black/white (horizontal)": black_white_horizontal_factory,
    "Black/white (vertical)": black_white_vertical_factory,
    "Cat": cat_factory,
    "Coins": coins_factory,
}

ui.page_opts(title="Edge Detection via Derivatives", fillable=True)

with ui.sidebar():
    ui.input_selectize(
        "image",
        "Image",
        list(IMAGE_FACTORIES),
    )
    ui.input_slider("threshold", "Threshold", 0.0, 4.0, 2.0)


@reactive.calc
def img():
    factory = IMAGE_FACTORIES[input.image()]
    return factory()


@reactive.calc
def grad_x():
    return convolve2d(GRAD_X, img(), mode="valid")


@reactive.calc
def grad_y():
    return convolve2d(GRAD_Y, img(), mode="valid")


@reactive.effect
def adjust_threshold_slider_bounds():
    grad_norm = np.sqrt(grad_x() ** 2 + grad_y() ** 2)
    min_grad = grad_norm.min()
    max_grad = grad_norm.max()

    ui.update_slider(
        "threshold",
        min=round(min_grad, 2),
        max=round(max_grad, 2),
        value=(min_grad + max_grad) / 2,
    )


with ui.layout_columns(col_widths=(6, 6)):
    with ui.card():

        @render.plot
        def image_2d():
            plt.imshow(img(), cmap="gray")
            plt.title("Original image")

    with ui.card():

        @render.plot
        def image_2d_edges():
            grad_norm = np.sqrt(grad_x() ** 2 + grad_y() ** 2)
            y_true, x_true = np.nonzero(grad_norm > input.threshold())

            plt.imshow(img(), cmap="gray")
            plt.plot(x_true, y_true, "r.", markersize=1)
            plt.title("Image with edges (red)")

    with ui.card():

        @render.plot
        def grad_x_plot():
            plt.imshow(grad_x())
            plt.title("Gradient in the x-direction")

    with ui.card():

        @render.plot
        def grad_y_plot():
            plt.imshow(grad_y())
            plt.title("Gradient in the y-direction")
