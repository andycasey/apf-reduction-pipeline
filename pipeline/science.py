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



class ScienceFrame(SpectroscopicFrame):


    def fit_apertures(self, index=None, profile="gaussian", **kwargs):
        """

        Some kwargs:

        sigma_detect_threshold [default 1]
        sigma_remove_threshold [default 3]
        """

        index = self.data.shape[0]/2 if index is None else int(index)

        # Take a slice down the middle and identify all the peak points
        aperture_midpoints = self._identify_initial_apertures(index,
            **kwargs)
        aperture_width = np.median(np.diff(aperture_midpoints))

        # Fit the apertures and refine the midpoints.
        apertures = []
        for midpoint in aperture_midpoints:
            apertures.append(self._fit_aperture(index, midpoint,
                aperture_width, profile))

        order = kwargs.get("projection_poly_order", 3)
        min_aperture_amplitude = kwargs.get("min_aperture_amplitude", 20)
        min_aperture_width = kwargs.get("min_aperture_width", 0.5)

        # Shake it to the left.
        added_left, skip = 1, 1
        while True:

            # Project out to the locations of additional apertures
            aperture_peaks = [_.mean.value for _ in apertures]
            aperture_offsets = np.diff(aperture_peaks)
            coeffs = np.polyfit(aperture_peaks[:-1], aperture_offsets, order)

            aperture_width_guess = np.polyval(coeffs, apertures[0].mean)
            midpoint_guess = apertures[0].mean - skip * aperture_width_guess

            logger.debug("{0} apertures and left-side aperture width guess is"\
                " {1:.0f}, giving mid-point guess of {2:.0f}".format(
                    len(apertures), aperture_width_guess, midpoint_guess))

            if not (self.shape[1] > midpoint_guess > 0):
                # We're done here.
                break

            aperture = self._fit_aperture(index, midpoint_guess,
                aperture_width_guess, profile)

            logger.debug("New aperture has b, amplitude, width = "\
                "{0:.1f}, {1:.1f}, {2:.1f}".format(aperture.b.value,
                    aperture.amplitude.value, aperture.width.value))

            if min_aperture_amplitude > aperture.amplitude \
            or min_aperture_width > aperture.width:
                skip += 1
                logger.debug("Skipping invalid aperture.")
                continue

            else:
                logger.debug("Added aperture on left side.")
                skip = 1
                apertures.insert(0, aperture)
                added_left += 1

        # Shake it to the right.
        added_right, skip = 1, 1
        while True:
            # Project out to the locations of additional apertures
            aperture_peaks = [_.mean.value for _ in apertures]
            aperture_offsets = np.diff(aperture_peaks)
            coeffs = np.polyfit(aperture_peaks[:-1], aperture_offsets, order)

            aperture_width_guess = np.polyval(coeffs, apertures[-1].mean)
            midpoint_guess = apertures[-1].mean + skip * aperture_width_guess

            logger.debug("{0} apertures and right-side aperture width guess is"\
                " {1:.0f}, giving mid-point guess of {2:.0f}".format(
                    len(apertures), aperture_width_guess, midpoint_guess))

            if not (self.shape[1] > midpoint_guess > 0):
                # We're done here.
                break

            aperture = self._fit_aperture(index, midpoint_guess,
                aperture_width_guess, profile)

            logger.debug("New aperture has b, amplitude, width = "\
                "{0:.1f}, {1:.1f}, {2:.1f}".format(aperture.b.value,
                    aperture.amplitude.value, aperture.width.value))

            if min_aperture_amplitude > aperture.amplitude \
            or min_aperture_width > aperture.width:

                skip += 1
                logger.debug("Skipping invalid aperture.")
                continue

            else:
                logger.debug("Added aperture on right side.")
                skip = 1
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
            multiplied by the `aperture.width` and if the difference between
            successive rows is greater than `row_limit * aperture.width`
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
            multiplied by the `aperture.width` and if the difference between
            successive rows is greater than `row_limit * aperture.width`
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

        aperture_width = abs(aperture.width) * 2. * 3 # 3 sigma either side.
        aperture_position = np.nan * np.ones(self.data.shape[0])
        aperture_position[slice_index] = aperture.mean.value

        # Shake it up.
        differences = np.zeros(self.data.shape[0])
        for i in np.arange(slice_index + 1, self.data.shape[0]):
            previous_mean = aperture_position[i - 1]

            # Get max within some pixels
            k = map(int, np.round(np.clip([
                    previous_mean - abs(aperture.width),
                    previous_mean + abs(aperture.width) + 1
                ], 0, self.data.shape[1])))
            new_mean = np.argmax(self.data[i, k[0]:k[1]]) + k[0]
            
            differences[i] = new_mean - previous_mean
            if abs(differences[i]) >= row_limit * abs(aperture.width):
                aperture_position[i] = previous_mean
            else:
                aperture_position[i] = new_mean

        # Shake it down.
        for i in np.arange(0, slice_index)[::-1]:
            previous_mean = aperture_position[i + 1]

            # Get max within some pixels
            k = map(int, np.round(np.clip([
                    previous_mean - abs(aperture.width),
                    previous_mean + abs(aperture.width) + 1
                ], 0, self.data.shape[1])))
            new_mean = np.argmax(self.data[i, k[0]:k[1]]) + k[0]

            differences[i] = new_mean - previous_mean
            if abs(differences[i]) >= row_limit * aperture.width:
                aperture_position[i] = previous_mean
            else:
                aperture_position[i] = new_mean

        # Shake it off, shake it off.
        return np.polyfit(np.arange(aperture_position.size),
            aperture_position, 2)


    def _trace_aperture_by_fitting(self, aperture, slice_index=None,
        row_limit=1, profile="gaussian"):
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
            multiplied by the `aperture.width` and if the difference between
            successive rows is greater than `row_limit * aperture.width`
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

        aperture_width = aperture.width * 2. * 3 # 3 sigma either side.
        aperture_position = np.nan * np.ones(self.data.shape[0])
        aperture_position[slice_index] = aperture.mean.value

        # Shake it up.
        differences = np.zeros(self.data.shape[0])
        for i in np.arange(slice_index + 1, self.data.shape[0]):
            previous_mean = aperture_position[i - 1]

            row_aperture = self._fit_aperture(i, int(previous_mean),
                aperture_width, width=aperture.width, profile=profile)

            differences[i] = row_aperture.mean - previous_mean
            if abs(differences[i]) >= row_limit * aperture.width:
                aperture_position[i] = previous_mean
            else:
                aperture_position[i] = row_aperture.mean.value

        # Shake it down.
        for i in np.arange(0, slice_index)[::-1]:
            previous_mean = aperture_position[i + 1]

            row_aperture = self._fit_aperture(i, int(previous_mean),
                aperture_width, width=aperture.width, profile=profile)

            differences[i] = row_aperture.mean - previous_mean
            if abs(differences[i]) >= row_limit * aperture.width:
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
            multiplied by the `aperture.width` and if the difference between
            successive rows is greater than `row_limit * aperture.width`
            then the aperture peak is temporarily rate-limited. Set this value
            to be low if you are seeing traces "jump" between apertures.

        :type row_limit:
            float
        """

        outlier_sigma_threshold = kwargs.pop("outlier_sigma_threshold", 1.5)
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


        # Fit polynomials to those that are not outliers
        degree = 3
        m_coefficients = np.zeros((degree + 1, coefficients.shape[1]))
        for i in range(m_coefficients.shape[1]):
            m_coefficients[:, i] = np.polyfit(x[~outliers],
                coefficients[~outliers, i], degree)




        """
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
        """


        """

        # Only return apertures that are not outliers, or ones that could be 
        # corrected.
        fig, axes = plt.subplots(3)
        ok = ~outliers
        for i, ax in enumerate(axes):
            scale = np.std(coefficients[:, i])
            mu = np.median(coefficients[:, i])
            mu, scale = 0, 1
            ax.scatter(x[outliers], (coefficients[outliers, i] - mu)/scale, facecolor="r")
            ax.scatter(x[ok], (coefficients[ok, i] - mu)/scale, facecolor="k")
            #ax.scatter(x[corrected], (coefficients[corrected, i] - mu)/scale, facecolor="g")
            ax.plot(x, np.polyval(m_coefficients[:, i], x), c="r")
            #ax.set_ylim(coefficients[:, i].min(), coefficients[:, i].max())

        raise a
        """

        return np.vstack([np.polyval(m_coefficients[:,i], x) \
            for i in range(coefficients.shape[1])]).T

        coefficients = coefficients[corrected | ~outliers]
        if coefficients.shape[0] != len(apertures):
            logger.warn("Traced {0} apertures, expected to trace {1}".format(
                coefficients.shape[0], len(apertures)))
            raise a
        return coefficients
        

    def _aperture_mask(self, coefficients, widths, discretize="round"):

        coefficients = np.atleast_2d(coefficients)
        if isinstance(widths, (int, float)):
            widths = widths * np.ones(coefficients.shape[0])
        else:
            widths = np.array(widths)
            assert widths.size == coefficients.shape[0], ("Widths must be a "\
                "float or an array the same length as coefficients dim[0]")

        discretize = discretize.lower()
        if discretize not in ("round", "bounded"):
            raise ValueError("discretize must be either round or bounded")

        aperture_mask = np.zeros(self.shape, dtype=bool)
        y = np.arange(self.shape[0])
        for c, w in zip(coefficients, widths):

            x = np.polyval(c, y)
            if discretize == "round":
                xis = np.round([x - w, x + w + 1]).astype(int).T

            elif discretize == "bounded":
                xis = np.vstack([
                    np.floor(x - w),
                    np.ceil(x + w) + 1
                ]).astype(int).T

            for yi, xi in zip(y, xis):
                aperture_mask[yi, slice(*xi)] = True

        return aperture_mask


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

        if np.any(self.flags["overscan"]):
            logger.warn("Extracting science apertures even though there are "
                "overscan pixels in the data that have not been removed.")

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
        

    def plot_apertures(self, apertures, index=None):

        index = self.data.shape[0]/2 if index is None else int(index)

        data_slice = self.data[int(index), None].flatten()
        fig, ax = plt.subplots()

        ax.plot(data_slice, c="k")

        width = abs(np.median([_.width for _ in apertures]))
        for aperture in apertures:
            x = np.linspace(
                aperture.mean - 5 * width,
                aperture.mean + 5 * width,
                100)
            ax.plot(x, aperture(x), c="r")

        return fig


    def fit_aperture_widths(self, coefficients):
        """
        Calculate the aperture widths at each point along the aperture traces.

        """
        coefficients = np.atleast_2d(coefficients)

        y = np.arange(self.shape[0])
        for c in coefficients:
            x = np.polyval(c, y)

            # At each index in y, we want to fit an aperture to the peak at x.


            # We have the x and y coordinate of the peak of the trace.
            # Slice across x? axis and fit to those data.
            # Use the data +/- the approximate aperture width.

            # Fixed: x position, background,
            # Free: peak value, width.

            f = self.plot_aperture_trace(coefficients)

            raise a



    def plot_aperture_trace(self, coefficients, widths=0, ax=None, **kwargs):
        """
        Plot the aperture trace across the CCD from the coefficients provided.

        :param coefficients:
            The polynomial coefficients that trace the aperture across the CCD.

        :type coefficients:
            :class:`numpy.array`

        :param widths: [optional]
            The widths of the aperture trace (in pixels). If provided, then the 
            plot will show the widths of the aperture across the CCD.

        :type widths:
            float or :class:`numpy.array`
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

        coefficients = np.atleast_2d(coefficients)
        if isinstance(widths, (int, float)):
            widths = widths * np.ones(coefficients.shape[0])

        else:
            widths = np.array(widths)
            assert widths.size == coefficients.shape[0], ("Widths must be a "\
                "float or an array the same length as coefficients dim[0]")

        y = np.arange(self.data.shape[0])
        for c, w in zip(coefficients, widths):

            x = np.polyval(c, y)
            ax.plot(x, y, **kwargs)

            if w > 0:      
                _ = plot_kwargs.copy()
                _["linestyle"] = ":"
                ax.plot(x - w, y, **_)
                ax.plot(x + w, y, **_)

        ax.set_xlim(-0.5, self.shape[1] + 0.5)
        ax.set_ylim(-0.5, self.shape[0] + 0.5)

        return fig
