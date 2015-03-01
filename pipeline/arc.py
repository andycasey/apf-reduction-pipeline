# coding: utf-8

""" Class for dealing with arc frames. """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

# Standard library
import cPickle as pickle
import logging
from time import sleep

# Third-party
import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
from astropy.modeling.functional_models import custom_model_1d
from scipy.stats import linregress
from ccd import CCD


# Create logger
logger = logging.getLogger(__name__)


@custom_model_1d
def EmissionProfile(x, b=0., mean=0., stddev=1., amplitude=1.):
    return models.Linear1D.eval(x, slope=0., intercept=b) + \
           models.Gaussian1D.eval(x, mean=mean, stddev=stddev, \
               amplitude=amplitude)



def wavelength_calibration(pixel_dispersion_mapping, order=3):

    pixel_dispersion_mapping = np.atleast_2d(pixel_dispersion_mapping)
    if order >= pixel_dispersion_mapping.size:
        raise ValueError("insufficient points to build a pixel-dispersion map")

    _ = pixel_dispersion_mapping[:, 0].argsort()
    pixel_dispersion_mapping = pixel_dispersion_mapping[_]
    coefficients = np.polyfit(pixel_dispersion_mapping[:, 0],
        pixel_dispersion_mapping[:, 1], order)

    return lambda x: np.polyval(coefficients, x)



class WavelengthDispersionMapper(object):

    def __init__(self, thar_atlas, arc_spectra, science_spectra, ascending_orders=False,
        reverse_disp=True, degree=2):

        self.thar_atlas = thar_atlas
        self.arc_spectra = arc_spectra
        self.science_spectra = science_spectra
        self.reverse_disp = reverse_disp
        self.ascending_orders = ascending_orders # True means left-most order is blue.
        self.degree = degree
        self.max_degree = 2

        self.order = 0
        self.points = [[] for _ in range(len(arc_spectra))]


        self._initial_draw()
        

        # Left/right will show the next or previous order.
        # Click on the data axes to fit a line. Then push 'y' to use the fit or
        # 'n' not to:

        # y --> will wait for corresponding fit line in top axes. 'y'/'no' needed
        # n --> will clear the fit on the data.

        self._waiting = 0



    def _initial_draw(self):
        """ Draw the initial things on the figure. """
         
        self.figure, self.axes = plt.subplots(3)

        self.axes[0].plot(self.thar_atlas[:, 0], self.thar_atlas[:, 1], c="k")
        self.axes[0].set_xlim(self.thar_atlas[0, 0], self.thar_atlas[-1, 0])
        self.axes[0].set_ylim(0, self.axes[0].get_ylim()[1])
        self.axes[0].set_title("ThAr Atlas")

        # Empty data for the other figures.
        self.axes[1].plot([], c="k")
        self.axes[2].plot([], c="k")

        self.figure.canvas.draw()

        self._cids = [
            self.figure.canvas.mpl_connect("button_press_event", self._button_press_event),
            self.figure.canvas.mpl_connect("key_press_event", self._key_press_event)
        ]

        self._redraw()

        return None



    def _redraw(self):

        # Clear additional lines
        axes = (self.axes[0], self.axes[1])
        for ax in axes:
            for line in ax.lines[1:]:
                line.set_visible(False)
            del ax.lines[1:]


        # Re-draw the data
        y = self.arc_spectra[self.order]
        x = np.arange(y.size)



        _ = [1, -1][self.reverse_disp]
        self.axes[1].set_title("Order {0}/{1}".format(
            self.order + 1, len(self.arc_spectra)))


        points = np.array(self.points[self.order])
        if len(points) > 0:
            points = points[np.argsort(points[:, 0])]
            for p in points[:, 1]:
                self.axes[0].axvline(p, c='r', zorder=-100)

        print("points are {}".format(points))

        y = self.science_spectra[self.order]
        if len(points) < 2:
            self.axes[2].lines[0].set_data(x[::_], y)
            self.axes[2].set_xlim(x[0], x[-1])
            self.axes[2].set_ylim(0, 1.1 * np.nanmax(y))

            y = self.arc_spectra[self.order]
            self.axes[1].lines[0].set_data(x[::_], y)
            self.axes[1].set_xlim(x[0], x[-1])
            self.axes[1].set_ylim(0, 1.1 * np.nanmax(y))


        else:
            # Show the bottom axes in WL!
            print("Fitting with: {}".format(points))
            print("X is {}".format(x))

            coefficients = np.polyfit(points[:, 0], points[:, 1], self.max_degree)
            x_ = np.polyval(coefficients, x[::_])
            self._x = x_

            self.axes[2].lines[0].set_data(x_, y)
            self.axes[2].set_xlim(x_.min(), x_.max())
            self.axes[2].set_ylim(0, 1.1 * np.nanmax(y))

            self.axes[0].set_xlim(x_.min(), x_.max())

            y = self.arc_spectra[self.order]
            self.axes[1].lines[0].set_data(x_, y)
            self.axes[1].set_xlim(x_.min(), x_.max())
            self.axes[1].set_ylim(0, 1.1 * np.nanmax(y))

            for p in points[:, 0]:
                self.axes[1].axvline(np.polyval(coefficients, p), c='r', zorder=-100)


        self.figure.canvas.draw()

        return True



    def _button_press_event(self, event):
        print("_button_press_event", event.__dict__)

        # Assuming we are not waiting on information

        if event.inaxes == self.axes[1]:
            # Fit a line profile to the observed data.

            _ = [1, -1][self.reverse_disp]
            if len(np.array(self.points[self.order])) > 2:
                idx = self._x[::_].searchsorted(event.xdata)
                x = np.arange(self.arc_spectra[self.order].size)
                self._point = [x[idx]]
            else:
                self._point = [event.xdata]


            # Set that we are waiting on a corresponding atlas point
            print("OK WE WILL WAIT FOR THE ATLAS POINT for {}".format(self._point))
            self._waiting = 1

        elif event.inaxes == self.axes[0] and self._waiting == 1:
            self._point.append(event.xdata)

            self.points[self.order].append(self._point)

            print("Got it: {}".format(self.points[self.order]))
            self._waiting = 0
            self._point = []
            
            if len(self.points[self.order]) > self.degree:
                self._redraw()

        return True

    def _key_press_event(self, event):
        print("_key_press_event", event.__dict__)

        # Assuming we are not waiting on information

        if event.key in ("left", "right"):
            self.order = np.clip(self.order + [-1, +1][event.key == "right"], 0,
                len(self.arc_spectra) - 1)
            print("Updated order to {}".format(self.order))
            self._redraw()

        if event.key == "d":
            print("Deleting the points for this order")
            self.points[self.order] = []
            self._redraw()

        return True





class ArcFrame(CCD):

    pass
