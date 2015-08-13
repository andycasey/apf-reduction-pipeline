# coding: utf-8

""" Class for dealing with generic spectroscopy frames. """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

# Standard library
import logging

# Third-party
from astropy.modeling import models, fitting
from astropy.modeling.functional_models import custom_model_1d
import numpy as np

from ccd import CCD

# Create logger
logger = logging.getLogger(__name__)


@custom_model_1d
def ApertureProfile(x, b=0., mean=0., width=1., amplitude=1.):
    return models.Linear1D.evaluate(x, slope=0., intercept=b) + \
           models.Gaussian1D.evaluate(x, mean=mean, stddev=width, \
               amplitude=amplitude)

@custom_model_1d
def BoxProfile(x, b=0., mean=0., width=1., amplitude=1.):
    return models.Linear1D.evaluate(x, slope=0., intercept=b) + \
        models.Box1D.evaluate(x, x_0=mean, width=width, amplitude=amplitude)


class SpectroscopicFrame(CCD):
    """
    This class inherits from CCD, but has spectroscopy-related functionality.
    """

    def _identify_initial_apertures(self, slice_index, **kwargs):
        """
        Make an initial guess of the positions of apertures in the frame.

        :param slice_index:
            The slice_index of data frame to search along.

        :type slice_index:
            int

        :returns:
            Initial sub-pixel guesses of the aperture mid-points.

        :rtype:
            :class:`numpy.array`
        """

        # Slice the data at the correct point
        data_slice = self.data[int(slice_index), None].flatten()

        # Assert something about the data slice shape?
        # [TODO]


        # Identify regions above some threshold.
        sigma_detect_threshold = kwargs.pop("sigma_detect_threshold", 1)
        sigma = (data_slice - np.nanmedian(data_slice))/np.nanstd(data_slice)
        indices = np.where(sigma > sigma_detect_threshold)[0]

        # Group neighbouring pixels and find the pixel mid-point of those groups
        groups = np.array_split(indices, np.where(np.diff(indices) != 1)[0] + 1)
        aperture_midpoints = np.array(map(np.mean, groups))
        
        sigma_remove_threshold = kwargs.pop("sigma_remove_threshold", 3)
        while True:
            # Remove outliers
            aperture_sep = np.diff(aperture_midpoints)
            median_aperture_sep = np.median(aperture_sep)

            sigmas = (aperture_sep - median_aperture_sep)/np.std(aperture_sep)
            indices = sigma_remove_threshold > np.abs(sigmas)
            if np.all(indices):
                break
            aperture_midpoints = aperture_midpoints[indices]


        if False:
        
            # We might have missed some order, so get the median separation.
            missed_aperture_threshold = kwargs.pop("missed_aperture_threshold", 1)
            sigmas = (aperture_sep - median_aperture_sep)/np.std(aperture_sep)
            missed_indices = np.where(sigmas > missed_aperture_threshold)[0]

            # Add in the ones we missed. For each point we want to approximately
            # how many nearby ones were missed. For example, we could miss 3 in a
            # row, so we should know about that.
            for index in missed_indices:
                distance = np.round(aperture_sep[index] / median_aperture_sep)
                aperture_midpoints = np.append(aperture_midpoints, np.arange(
                    aperture_midpoints[index], aperture_midpoints[index+1], 
                    aperture_sep[index] / distance)[1:])

        return aperture_midpoints



    def _fit_aperture(self, slice_index, peak_index, aperture_width, profile,
        **p0):
        """
        Fit an aperture slice with a Gaussian profile and some background.

        :param slice_index:
            The index to slice across the axis.

        :type slice_index:
            int

        :param peak_index:
            The approximate sub-pixel point where the aperture peak is located.

        :type peak_index:
            float

        :param aperture_width:
            The approximate width between aperture peaks in pixels.

        :type aperture_width:
            float
        """

        indices = (
            peak_index - aperture_width/2.,
            peak_index + aperture_width/2. + 1
        )
        indices = np.clip(map(int, [np.floor(indices[0]), np.ceil(indices[1])]),
            0, self.data.shape[1])
        
        # Get the data.
        x = np.arange(*indices)
        y = self.data[int(slice_index), indices[0]:indices[1]].flatten()

        if profile.lower() == "gaussian":
            profile_shape = ApertureProfile()

        elif profile.lower() == "box":
            profile_shape = BoxProfile()

        else:
            raise ValueError("unknown profile type given: {}".format(profile))

        # The logic for the initial guess of stddev is as follows:
        # aperture_width is the approximate full width between this aperture
        # and the next. Therefore this width encapsulates ~5 sigma either side
        # of the line. So stddev is ~aperture_width/10.
        default_p0 = dict(b=min(y), amplitude=max(y), width=aperture_width/10.,
            mean=np.mean(x))
        for k, v in default_p0.items():
            setattr(profile_shape, k, p0.get(k, v))

        fit = fitting.LevMarLSQFitter()
        return fit(profile_shape, x, y)


