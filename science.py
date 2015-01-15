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



class ScienceFrame(SpectroscopicFrame):


    def trace_aperture(self, aperture, slice_index=None, row_rate_limit=1):
        """
        Trace an aperture along the CCD.

        :param aperture:
            The aperture fit from a slice on the CCD.

        :type aperture:
            :class:`~astropy.modeling.Model`

        :param slice_index: [optional]
            The index that the aperture was measured along. If not provided, the
            aperture is assumed to be measured at the mid-plane of the CCD.

        :type slice_index:
            int

        :param row_rate_limit: [optional]
            Limit the difference between column aperture fits. This value is 
            multiplied by the `aperture.stddev` and if the difference between
            successive rows is greater than `row_rate_limit * aperture.stddev`
            then the aperture peak is temporarily rate-limited. Set this value
            to be low if you are seeing traces "jump" between apertures.

        :type row_rate_limit:
            float
        """

        # If no slice index was provided, let's assume it was measured at the
        # CCD mid-plane.
        slice_index = self.data.shape[0]/2 if slice_index is None \
            else int(slice_index)

        row_rate_limit = float(row_rate_limit)
        if 0 >= row_rate_limit:
            raise ValueError("row rate limit must be positive")

        aperture_width = aperture.stddev * 2. * 3 # 3 sigma either side.
        aperture_position = np.nan * np.ones(self.data.shape[0])
        aperture_position[slice_index] = aperture.mean.value

        # Shake it up.
        differences = np.zeros(self.data.shape[0])
        for i in np.arange(slice_index + 1, self.data.shape[0]):
            previous_mean = aperture_position[i - 1]

            x, row_aperture = self._fit_aperture(i, int(previous_mean),
                aperture_width, stddev=aperture.stddev)

            differences[i] = row_aperture.mean - previous_mean
            if abs(differences[i]) >= row_rate_limit * aperture.stddev:
                aperture_position[i] = previous_mean
            else:
                aperture_position[i] = row_aperture.mean.value

        # Shake it down.
        for i in np.arange(0, slice_index)[::-1]:
            previous_mean = aperture_position[i + 1]

            x, row_aperture = self._fit_aperture(i, int(previous_mean),
                aperture_width, stddev=aperture.stddev)

            differences[i] = row_aperture.mean - previous_mean
            if abs(differences[i]) >= row_rate_limit * aperture.stddev:
                aperture_position[i] = previous_mean
            else:
                aperture_position[i] = row_aperture.mean.value

        return aperture_position


    def trace_apertures(self):

        # Fit aperture at the mid-plane.
        index = self.data.shape[0]/2
        apertures = self.fit_apertures(index)


        # Move the trace up and down, searching within nearby pixels.

        num_apertures = len(apertures)
        aperture_means = np.nan*np.ones((self.data.shape[0], num_apertures))
        aperture_means[index, :] = [_[1].mean.value for _ in apertures]

        aperture_width = np.diff(aperture_means[index, :]).min()

        # Shake it down.
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
        ax[0].imshow(self.data, cmap=plt.cm.Greys_r, vmin=0, vmax=1500, aspect="auto")
        ax = ax[1]
        ax.imshow(self.data, cmap=plt.cm.Greys_r, vmin=0, vmax=1500, aspect="auto")
        ax.set_color_cycle(["r", "b"])

        rate_limit = 2.5 # pixels per rate_pixels
        rate_pixels = 10

        for i in range(num_apertures):

            for j in np.arange(0, index)[::-1]:

                previous_mean = aperture_means[j + 1, i]
                print("j = {0}, previous mean was {1}".format(j, previous_mean))

    
                k = np.arange(self.data[j].size).searchsorted([
                    previous_mean - aperture_width/4.,
                    previous_mean + aperture_width/4.])

                new_mean = np.argmax(self.data[j, k[0]:k[1]]) + k[0]

                # rate limit the mean?
                rate = abs(np.nansum(np.diff(aperture_means[j:j+rate_pixels, i])))
                print("RATE {0}".format(rate))
                if rate >= rate_limit:
                    new_mean = previous_mean

                print("found max at {0}".format(new_mean))

                if abs(new_mean - previous_mean) > aperture_width/4.:
                    raise WTFError()

                print("difference was {0}".format(new_mean - previous_mean))

                aperture_means[j, i] = new_mean

            # Shake it up.
            for j in np.arange(index+1, self.data.shape[0]):

                previous_mean = aperture_means[j - 1, i]
                print("j = {0}, previous mean was {1}".format(j, previous_mean))

                k = np.arange(self.data[j].size).searchsorted([
                    previous_mean - aperture_width/4.,
                    previous_mean + aperture_width/4.])

                print("PIXELS {0}".format(k.size))

                new_mean = np.argmax(self.data[j, k[0]:k[1]]) + k[0]
                print("found max at {0}".format(new_mean))

                difference = abs(new_mean - previous_mean)
                if difference > aperture_width/4.:
                    raise WTFError()

                print("difference was {0}".format(difference))

                aperture_means[j, i] = new_mean


            line = aperture_means[:, i]
            ax.plot(line, np.arange(line.size), c="b")
            ax.axhline(index, c='k')
            plt.show()

        ax.set_xlim(-0.5, self.data.shape[1] + 0.5)
        ax.set_ylim(-0.5, self.data.shape[0] + 0.5)

        raise a



    def fit_apertures(self, index):

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
            aperture_peaks = [_[1].mean.value for _ in apertures]
            aperture_offsets = np.diff(aperture_peaks)
            coeffs = np.polyfit(aperture_peaks[:-1], aperture_offsets, 2)

            aperture_width_guess = np.polyval(coeffs, apertures[0][1].mean)
            if aperture_width_guess < 0: break

            midpoint_guess = apertures[0][1].mean - aperture_width_guess
            x, aperture = self._fit_aperture(index, midpoint_guess,
                aperture_width_guess)

            if aperture.b > aperture.amplitude: break
            apertures.insert(0, (x, aperture))
            added_left += 1

        # Shake it to the right.
        added_right = 1
        while True:
            # Project out to the locations of additional apertures
            aperture_peaks = [_[1].mean.value for _ in apertures]
            aperture_offsets = np.diff(aperture_peaks)
            coeffs = np.polyfit(aperture_peaks[:-1], aperture_offsets, 2)

            aperture_width_guess = np.polyval(coeffs, apertures[-1][1].mean)
            if aperture_width_guess < 0: break

            midpoint_guess = apertures[-1][1].mean + aperture_width_guess
            x, aperture = self._fit_aperture(index, midpoint_guess,
                aperture_width_guess)

            if aperture.b > aperture.amplitude: break
            apertures.append((x, aperture))
            added_right += 1

        # Shake it all about.
        return apertures


    def extract_apertures(self, wavelength_calibration=None):
        raise NotImplementedError


