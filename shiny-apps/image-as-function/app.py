from shiny import reactive
from shiny.express import input, render, ui
from shinywidgets import render_plotly

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from skimage import data
from skimage.color import rgb2gray
from skimage.filters import gaussian

ui.page_opts(fillable=True)


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




ui.page_opts(title="Images as functions", fillable=True)

ui.input_selectize(
    "image", "Image",
    list(IMAGE_FACTORIES),
)

@reactive.calc
def img():
    factory = IMAGE_FACTORIES[input.image()]
    return factory()

with ui.layout_columns(col_widths=(6, 6)):
    with ui.card():
        @render.plot
        def image_2d_next_panel():
            plt.imshow(img(), cmap="gray")
            plt.title("Original image")

    with ui.card():
        @render_plotly
        def image_3d():
            fig = go.Figure(data=[go.Surface(z=img())])
            fig.update_layout(
                autosize=False,
                width=500, height=500,
                margin=dict(l=65, r=50, b=65, t=90))
            return fig
