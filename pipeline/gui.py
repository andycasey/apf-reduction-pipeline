# coding: utf-8

""" A poor man's GUI for the reduction pipeline. """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

# Standard library
import logging

# Third-party
import numpy as np
import matplotlib.pyplot as plt

from spectroscopy import SpectroscopicFrame

# Create logger
logger = logging.getLogger(__name__)




def interactively_fit_apertures(frame, vmin=0, vmax=1500, axis=0):

    assert axis in (0, 1)

    fig, (image_ax, slice_ax) = plt.subplots(ncols=2)

    image_ax.imshow(frame.data, cmap=plt.cm.Greys_r, vmin=vmin, vmax=vmax,
        aspect="auto")
    image_ax.xaxis.set_visible(False)
    image_ax.yaxis.set_visible(False)

    plot_apertures = []
    plot_data = slice_ax.plot([], c="k")

    def onclick(event):

        if event.inaxes != image_ax: return

        print("Fitting")
        if axis == 0:
            index = int(event.ydata)
            y = frame.data[index, :].flatten()
            
        else:
            index = int(event.xdata)
            y = frame.data[:, index].flatten()

        x = np.arange(y.size)
        plot_data[0].set_data(np.vstack([x, y]))
        slice_ax.set_xlim(0, y.size)
        slice_ax.set_ylim(y.min(), y.max())

        if axis == 0:
            apertures = obs.fit_apertures(index)

            if len(apertures) > len(plot_apertures):
                plot_apertures.extend([slice_ax.plot([], c="b")[0] \
                    for _ in range(len(apertures) - len(plot_apertures))])

            for i, aperture in enumerate(apertures):
                x = np.arange(aperture.mean - 10, aperture.mean + 10, 0.01)
                plot_apertures[i].set_data(np.vstack([x, aperture(x)]))

            empty_data = np.nan * np.ones((2,2))
            [_.set_data(empty_data) for _ in plot_apertures[i+1:]]

        fig.canvas.draw()

        print('button={0}, x={1}, y={2}, xdata={3}, ydata={4}'.format(
            event.button, event.x, event.y, event.xdata, event.ydata))


    cid = fig.canvas.mpl_connect('button_press_event', onclick)

