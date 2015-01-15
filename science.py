# coding: utf-8

""" Class for dealing with science frames. """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

# Standard library
import logging

# Third-party
from astropy.modeling import models, fitting
from astropy.modeling.functional_models import custom_model_1d
import numpy as np


from spectroscopy import SpectroscopicFrame

# Create logger
logger = logging.getLogger(__name__)


@custom_model_1d
def ApertureProfile(x, b=0., mean=0., stddev=0.05, amplitude=1.):
    return models.Linear1D.eval(x, slope=0., intercept=b) + \
           models.Gaussian1D.eval(x, mean=mean, stddev=stddev, \
                amplitude=amplitude)

"""


        # Now let's fit the actual apertures to refine the peak points and then
        # we will gradually grow the apertures outwards until we can't find any
        # more.
        d = int(aperture_midpoints[0])
        indices = (d - int(median_aperture_sep)/2,
            d + int(median_aperture_sep)/2 + 1)

        x = np.arange(*indices)
        y = data_slice[slice(*indices)]

        g = ApertureProfile()
        default_p0 = dict(b=min(y), amplitude=max(y), stddev=1., mean=np.mean(x))

        for k, v in default_p0.items():
            setattr(g, k, v)

        fit = fitting.LevMarLSQFitter()
        something = fit(g, x, y)


        # Move outwards to identify other potential apertures that didn't meet
        # our initial threshold.
        grow = kwargs.pop("num_apertures_grow", 3)
        aperture_midpoints = np.sort(aperture_midpoints)
        aperture_sep = np.diff(aperture_midpoints)

        lhs_midpoints = np.arange(aperture_midpoints[0] % aperture_sep[0], 
            aperture_midpoints[0], aperture_sep[0])[-grow:]

        rhs_midpoints = np.arange(aperture_midpoints[-1] + aperture_sep[-1],
            data_slice.size, aperture_sep[-1])[:grow]

        possible_midpoints = np.append(lhs_midpoints, rhs_midpoints)

        # Exclude the midpoint range that we already have
        #exclude = (aperture_midpoints[-1] >= possible_midpoints) \
        #    * (possible_midpoints >= aperture_midpoints[0])
        #possible_midpoints = possible_midpoints[~exclude]

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(data_slice, 'k')

        for midpoint in aperture_midpoints:
            ax.axvline(midpoint, c='b')

        for possible_midpoint in possible_midpoints:
            ax.axvline(possible_midpoint, c='r')

        ax.plot(x, y, c='g')

        ax.set_xlim(0, data_slice.size)
        ax.set_ylim(0, data_slice.max())

        plt.show()
        raise a

"""



class ScienceFrame(SpectroscopicFrame):


    def trace_apertures(self, axis=0):
        # trace orders -- interactively?

        # Take a slice down the middle and identify all the peak points
        index = self.data.shape[axis]/2
        aperture_midpoints = self._identify_initial_apertures(axis, index)
        aperture_width = np.diff(aperture_midpoints).mean()

        # Fit the apertures and refine the midpoints.
        apertures = []
        for midpoint in aperture_midpoints:
            apertures.append(self._fit_aperture(axis, index, midpoint,
                aperture_width))



        import matplotlib.pyplot as plt

        full_slice = [int(index) if i == axis else None \
            for i in range(len(self.data.shape))]
        data_slice = self.data[full_slice].flatten()

        fig, ax = plt.subplots()
        ax.plot(data_slice, 'k')

        for midpoint, aperture in zip(aperture_midpoints, apertures):
            ax.axvline(midpoint, c='b')

            # Get some data around here
            i = int(midpoint)
            i = (i - int(aperture_width), i + int(aperture_width) + 1)
            x = np.arange(*i)
            ax.plot(x, aperture(x), c='m')


        ax.set_xlim(0, data_slice.size)
        ax.set_ylim(0, data_slice.max())

        plt.show()
        raise a

        # Now step outwards and identify new 


        # They should be somewhat periodically spaced, or at most linearly
        # periodic. We need to work out the periodicity and use that to identify
        # the peaks of the apertures. 

        # Once we have the peaks of the apertures at the mid-plane, we should
        # bisect the rows and repeatedly do this so that we fully map out how
        # the apertures move across the CCD

        # When we have the peak apertures at N given points, we should fit
        # polynomials to all the aperture lines and draw them.




        raise NotImplementedError


    def extract_apertures(self, wavelength_calibration=None):
        raise NotImplementedError


