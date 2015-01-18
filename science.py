# coding: utf-8

""" Class for dealing with science frames. """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

# Standard library
import logging

# Third-party
import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
from astropy.modeling.functional_models import custom_model_1d
from scipy.interpolate import splrep, splev
from scipy.stats import linregress

from spectroscopy import SpectroscopicFrame

# Create logger
logger = logging.getLogger(__name__)


@custom_model_1d
def ApertureProfile(x, b=0., mean=0., stddev=0.05, amplitude=1.):
    return models.Linear1D.eval(x, slope=0., intercept=b) + \
           models.Gaussian1D.eval(x, mean=mean, stddev=stddev, \
                amplitude=amplitude)


class ScienceFrame(SpectroscopicFrame):


    def fit_apertures(self, index=None):
        

        index = self.data.shape[0]/2 if index is None else int(index)

        # Take a slice down the middle and identify all the peak points
        aperture_midpoints = self._identify_initial_apertures(index)
        aperture_width = np.median(np.diff(aperture_midpoints))

        # Fit the apertures and refine the midpoints.
        apertures = []
        for midpoint in aperture_midpoints:
            apertures.append(self._fit_aperture(index, midpoint,
                aperture_width))

        # Shake it to the left.
        added_left = 1
        while True:

            # Project out to the locations of additional apertures
            aperture_peaks = [_.mean.value for _ in apertures]
            aperture_offsets = np.diff(aperture_peaks)
            coeffs = np.polyfit(aperture_peaks[:-1], aperture_offsets, 2)

            aperture_width_guess = np.polyval(coeffs, apertures[0].mean)
            if aperture_width_guess < 0: break

            midpoint_guess = apertures[0].mean - aperture_width_guess
            aperture = self._fit_aperture(index, midpoint_guess,
                aperture_width_guess)

            if aperture.b > aperture.amplitude: break
            apertures.insert(0, aperture)
            added_left += 1

        # Shake it to the right.
        added_right = 1
        while True:
            # Project out to the locations of additional apertures
            aperture_peaks = [_.mean.value for _ in apertures]
            aperture_offsets = np.diff(aperture_peaks)
            coeffs = np.polyfit(aperture_peaks[:-1], aperture_offsets, 2)

            aperture_width_guess = np.polyval(coeffs, apertures[-1].mean)
            if aperture_width_guess < 0: break

            midpoint_guess = apertures[-1].mean + aperture_width_guess
            aperture = self._fit_aperture(index, midpoint_guess,
                aperture_width_guess)

            if aperture.b > aperture.amplitude: break
            apertures.append(aperture)
            added_right += 1

        # Shake it all about.
        return apertures


    def trace_aperture(self, aperture, slice_index=None, row_limit=1,
        method="fast", **kwargs):
        """
        Trace an aperture along the CCD. If the 'fast' method is used (default)
        then apertures are found by identifying the maximum (nearby) pixel value
        in each row. If the 'slow' method is used, then Gaussian profiles are
        fit to each row.

        :param aperture:
            The aperture fit from a slice on the CCD.

        :type aperture:
            :class:`~astropy.modeling.Model`

        :param slice_index: [optional]
            The index that the aperture was measured along. If not provided, the
            aperture is assumed to be measured at the mid-plane of the CCD.

        :type slice_index:
            int

        :param row_limit: [optional]
            Limit the difference between column aperture fits. This value is 
            multiplied by the `aperture.stddev` and if the difference between
            successive rows is greater than `row_limit * aperture.stddev`
            then the aperture peak is temporarily rate-limited. Set this value
            to be low if you are seeing traces "jump" between apertures.

        :type row_limit:
            float
        """

        method = method.lower()
        if method not in ("fast", "slow"):
            raise ValueError("method must be fast or slow")

        func = self._trace_aperture_by_max if method == "fast" \
            else self._trace_aperture_by_fitting

        return func(aperture, slice_index, row_limit, **kwargs)


    def _trace_aperture_by_max(self, aperture, slice_index=None, row_limit=1):
        """
        Trace an aperture along the CCD by finding the maximal pixel value in
        some small region.

        :param aperture:
            The aperture fit from a slice on the CCD.

        :type aperture:
            :class:`~astropy.modeling.Model`

        :param slice_index: [optional]
            The index that the aperture was measured along. If not provided, the
            aperture is assumed to be measured at the mid-plane of the CCD.

        :type slice_index:
            int

        :param row_limit: [optional]
            Limit the difference between column aperture fits. This value is 
            multiplied by the `aperture.stddev` and if the difference between
            successive rows is greater than `row_limit * aperture.stddev`
            then the aperture peak is temporarily rate-limited. Set this value
            to be low if you are seeing traces "jump" between apertures.

        :type row_limit:
            float
        """

        # If no slice index was provided, let's assume it was measured at the
        # CCD mid-plane.
        slice_index = self.data.shape[0]/2 if slice_index is None \
            else int(slice_index)

        row_limit = float(row_limit)
        if 0 >= row_limit:
            raise ValueError("row rate limit must be positive")

        aperture_width = abs(aperture.stddev) * 2. * 3 # 3 sigma either side.
        aperture_position = np.nan * np.ones(self.data.shape[0])
        aperture_position[slice_index] = aperture.mean.value

        # Shake it up.
        differences = np.zeros(self.data.shape[0])
        for i in np.arange(slice_index + 1, self.data.shape[0]):
            previous_mean = aperture_position[i - 1]

            # Get max within some pixels
            k = map(int, np.round(np.clip([
                    previous_mean - abs(aperture.stddev),
                    previous_mean + abs(aperture.stddev) + 1
                ], 0, self.data.shape[1])))
            new_mean = np.argmax(self.data[i, k[0]:k[1]]) + k[0]
            
            differences[i] = new_mean - previous_mean
            if abs(differences[i]) >= row_limit * abs(aperture.stddev):
                aperture_position[i] = previous_mean
            else:
                aperture_position[i] = new_mean

        # Shake it down.
        for i in np.arange(0, slice_index)[::-1]:
            previous_mean = aperture_position[i + 1]

            # Get max within some pixels
            k = map(int, np.round(np.clip([
                    previous_mean - abs(aperture.stddev),
                    previous_mean + abs(aperture.stddev) + 1
                ], 0, self.data.shape[1])))
            new_mean = np.argmax(self.data[i, k[0]:k[1]]) + k[0]

            differences[i] = new_mean - previous_mean
            if abs(differences[i]) >= row_limit * aperture.stddev:
                aperture_position[i] = previous_mean
            else:
                aperture_position[i] = new_mean

        # Shake it off, shake it off.
        return np.polyfit(np.arange(aperture_position.size),
            aperture_position, 2)


    def _trace_aperture_by_fitting(self, aperture, slice_index=None,
        row_limit=1):
        """
        Trace an aperture along the CCD by fittin Gaussian profiles at every
        row.

        :param aperture:
            The aperture fit from a slice on the CCD.

        :type aperture:
            :class:`~astropy.modeling.Model`

        :param slice_index: [optional]
            The index that the aperture was measured along. If not provided, the
            aperture is assumed to be measured at the mid-plane of the CCD.

        :type slice_index:
            int

        :param row_limit: [optional]
            Limit the difference between column aperture fits. This value is 
            multiplied by the `aperture.stddev` and if the difference between
            successive rows is greater than `row_limit * aperture.stddev`
            then the aperture peak is temporarily rate-limited. Set this value
            to be low if you are seeing traces "jump" between apertures.

        :type row_limit:
            float
        """

        # If no slice index was provided, let's assume it was measured at the
        # CCD mid-plane.
        slice_index = self.data.shape[0]/2 if slice_index is None \
            else int(slice_index)

        row_limit = float(row_limit)
        if 0 >= row_limit:
            raise ValueError("row rate limit must be positive")

        aperture_width = aperture.stddev * 2. * 3 # 3 sigma either side.
        aperture_position = np.nan * np.ones(self.data.shape[0])
        aperture_position[slice_index] = aperture.mean.value

        # Shake it up.
        differences = np.zeros(self.data.shape[0])
        for i in np.arange(slice_index + 1, self.data.shape[0]):
            previous_mean = aperture_position[i - 1]

            row_aperture = self._fit_aperture(i, int(previous_mean),
                aperture_width, stddev=aperture.stddev)

            differences[i] = row_aperture.mean - previous_mean
            if abs(differences[i]) >= row_limit * aperture.stddev:
                aperture_position[i] = previous_mean
            else:
                aperture_position[i] = row_aperture.mean.value

        # Shake it down.
        for i in np.arange(0, slice_index)[::-1]:
            previous_mean = aperture_position[i + 1]

            row_aperture = self._fit_aperture(i, int(previous_mean),
                aperture_width, stddev=aperture.stddev)

            differences[i] = row_aperture.mean - previous_mean
            if abs(differences[i]) >= row_limit * aperture.stddev:
                aperture_position[i] = previous_mean
            else:
                aperture_position[i] = row_aperture.mean.value

        # Shake it off, shake it off.
        return np.polyfit(np.arange(aperture_position.size),
            aperture_position, 2)


    def trace_apertures(self, apertures, **kwargs):
        """
        Trace apertures across the CCD. There are two ways to do this

        :param apertures:
            The apertures fitted to a single slice.

        :type apertures:
            list of :class:`~astropy.modeling.Model`

        :param slice_index: [optional]
            The index that the aperture was measured along. If not provided, the
            aperture is assumed to be measured at the mid-plane of the CCD.

        :type slice_index:
            int

        :param row_limit: [optional]
            Limit the difference between column aperture fits. This value is 
            multiplied by the `aperture.stddev` and if the difference between
            successive rows is greater than `row_limit * aperture.stddev`
            then the aperture peak is temporarily rate-limited. Set this value
            to be low if you are seeing traces "jump" between apertures.

        :type row_limit:
            float
        """

        outlier_sigma_threshold = kwargs.pop("outlier_sigma_threshold", 2)
        coefficients = np.array([self.trace_aperture(_, **kwargs) \
            for _ in apertures])

        # Some of these may be not like the others,..
        # The implied assumption here is that the second-order curvature grows
        # linearly with pixels. And that we will have outliers.

        x = np.array([_.mean.value for _ in apertures])
        y = coefficients[:, 0]
        outliers = np.zeros(x.size, dtype=bool)
        while True:
            m, b, r_value, p_value, stderr = linregress(x[~outliers],
                y[~outliers])

            model = m * x[~outliers] + b
            sigma = (model - y[~outliers])/np.std(y[~outliers])
            new_outliers = np.abs(sigma) >= outlier_sigma_threshold
            if not np.any(new_outliers):
                break

            outliers[~outliers] += new_outliers

        # Should we recover outliers that are within the bounds?
        # Yes, yes we should.
        _, __ = np.where(~outliers)[0], np.where(outliers)[0]
        corrected_indices = __[(_.max() > __) * (__ > _.min())]
        corrected = np.zeros(outliers.size, dtype=bool)
        for index in corrected_indices:
            for j in range(coefficients.shape[1]):
                tck = splrep(x[~outliers], coefficients[~outliers, j])
                coefficients[index, j] = splev(x[index], tck)
            corrected[index] = True
            outliers[index] = True

        # Only return apertures that are not outliers, or ones that could be 
        # corrected.
        return coefficients[corrected | ~outliers]
        

    def extract_aperture(self, coefficients, width, discretize="round", 
        **kwargs):
        """
        Extract an aperture and return the flux.

        :param coefficients:
            The polynomial coefficients that trace the aperture across the CCD.

        :type coefficients:
            :class:`numpy.array`

        :param width:
            The width of the aperture trace (in pixels).

        :type width:
            float

        :param discretize: [optional]
            The behaviour to take when discretizing the pixels. Available
            behaviours are "round" (default) or "bounded". Rounded behaviour 
            will use a pixel if the rounded `mu + width` value is 1. Bounded
            behaviour will take the floor and ceiling of `mu - width` and `mu +
            width`, respectively

        :type discretize:
            str
        """

        if not np.any(self.flags["overscan"]):
            logger.warn("Extracting science apertures even though the overscan "
                "has not been removed.")

        y = np.arange(self.data.shape[0])
        x = np.polyval(coefficients, y)

        discretize = discretize.lower()
        if discretize == "round":
            x_indices = np.round([x - width, x + width + 1]).astype(int).T

        elif discretize == "bounded":
            x_indices = np.vstack([
                np.floor(x - width),
                np.ceil(x + width) + 1
            ]).astype(int).T

        else:
            raise ValueError("discretize must be either round or bounded")

        extracted_data = np.zeros(y.size)
        for yi, xi in zip(y, x_indices):
            extracted_data[yi] = self.data[yi, slice(*xi)].sum()
        return extracted_data
        

    def plot_aperture_trace(self, coefficients, width=0, ax=None, **kwargs):
        """
        Plot the aperture trace across the CCD from the coefficients provided.

        :param coefficients:
            The polynomial coefficients that trace the aperture across the CCD.

        :type coefficients:
            :class:`numpy.array`

        :param width: [optional]
            The width of the aperture trace (in pixels). If provided, then the 
            plot will show the width of the aperture across the CCD.

        :type width:
            float
        """

        if ax is None:
            fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)

            _ = self.data[self.flags["data"]].flatten()
            d = _[np.isfinite(_)]
            vmax = np.median(d) + 3 * np.std(d)
            _default = {
                "vmin": 0,
                "vmax": vmax,
                "aspect": "auto",
                "cmap": plt.cm.Greys_r
            }
            imshow_kwargs = kwargs.pop("imshow_kwargs", {})
            for k in set(_default.keys()).difference(imshow_kwargs.keys()):
                imshow_kwargs[k] = _default[k]
            [_.imshow(self.data, **imshow_kwargs) for _ in axes]
            ax = axes[1]

        else:
            fig = ax.figure

        y = np.arange(self.data.shape[0])
        x = np.polyval(coefficients, y)
        plot_kwargs = kwargs.copy()
        plot_kwargs.setdefault("c", "b")
        ax.plot(x, y, **plot_kwargs)

        if width > 0:      
            plot_kwargs["linestyle"] = ":"
            ax.plot(x - width, y, **plot_kwargs)
            ax.plot(x + width, y, **plot_kwargs)

        ax.set_xlim(-0.5, self.shape[1] + 0.5)
        ax.set_ylim(-0.5, self.shape[0] + 0.5)

        return fig
