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



class WavelengthSolutionFinder(object):


    def __init__(self, extracted_science_spectra, extracted_arc_spectra,
        thar_lines):

        assert len(extracted_arc_spectra) == len(extracted_science_spectra)

        self.thar_lines = thar_lines
        self.extracted_arc_spectra = extracted_arc_spectra
        self.extracted_science_spectra = extracted_science_spectra

        # Create a matplotlib interface
        self.figure, self.axes = plt.subplots(2)

        self.wavelength_points = [[] for _ in range(len(self.extracted_arc_spectra))]
        self.extraction_index = 0
        self.axes[0].plot([], c="k")
        self.axes[1].plot([], c="b")

        # Scatter plot for the extracted arc line points
        self._plotted_arc_lines = self.axes[1].scatter([], [])
        self.axes[0].set_xlabel("Pixel")

        self._redraw()

        self.figure.canvas.draw()


        # Set up callback
        self._cid = self.figure.canvas.mpl_connect("button_press_event",
            self.add_wavelength_point)

        self._cid2 = self.figure.canvas.mpl_connect("key_press_event",
            self._keypress)
        self._keys_pressed = ""
        self._waiting_for_keypress = False


    def _keypress(self, event):

        if self._waiting_for_keypress:
            
            print("Wavelength so far: {}".format(self._keys_pressed))

            if event.key == "enter":
                result = "" + self._keys_pressed
                self._keys_pressed = ""
                self._waiting_for_keypress = False
                self._keypress_callback(result)

            else:
                self._keys_pressed += event.key

        else:

            # Left/right to switch indices
            if event.key in ("left", "right"):

                offset = [-1, +1][event.key == "right"]
                self.extraction_index = np.clip(self.extraction_index + offset,
                    0, len(self.extracted_arc_spectra) - 1)

                print("Updated extraction index to {}".format(self.extraction_index))
                self._redraw()



    def _redraw(self):

        # Re-draw the data
        y = self.extracted_science_spectra[self.extraction_index]
        x = np.arange(y.size)
        self._x = x
        self.axes[0].lines[0].set_data(x, y)
        self.axes[0].set_xlim(x[0], x[-1])
        self.axes[0].set_ylim(0, np.nanmedian(y) + 2*np.nanstd(y))
        self.figure.canvas.draw()

        # Re-draw the current wavelength solution
        self.update_wavelength_solution()


    def add_wavelength_point(self, event):

        # Check that we are in the correct axes
        if event.inaxes != self.axes[0]:

            # We have been told to fit the nearest point to the nearest profile

            ax = self.axes[1]
            
            thar_lines_to_show = (self.thar_lines >= min(ax.get_xlim())) \
                * (max(ax.get_xlim()) >= self.thar_lines)

            thar_lines = self.thar_lines[thar_lines_to_show]


            distances = thar_lines - event.xdata
            print("distances", distances)

            thar_line_selected = thar_lines[np.argmin(np.abs(distances))]

            # Fit it to a peak
            data = self.extracted_arc_spectra[self.extraction_index]
            #self._x

            print("x2", self._x2)
            print("event", event)

            if self._x2[0] > self._x2[-1]:
                i = data.size - int(self._x2[::-1].searchsorted(event.xdata))

            else:
                i = int(self._x2.searchsorted(event.xdata))
            
            i = np.arange(data.size)[i]
            print("i", i)

            n = 25
            na, nb = np.clip(i - n, 0, data.size - 1), np.clip(i + n + 1, 0, data.size - 1)
            subset = np.argmax(data[na:nb])

            peak = subset + na
            peak = np.arange(data.size)[peak]

            # Is this line already in the list for this wl?
            existing_lines = np.array([_[1] for _ in self.wavelength_points[self.extraction_index]])
            if np.any(np.abs(thar_line_selected - existing_lines) < 0.01):

                # Update that one 
                print("updating existing arc line point")
                index = np.where(np.abs(thar_line_selected - existing_lines) < 0.01)[0]
                self.wavelength_points[self.extraction_index][index] = (peak, thar_line_selected)

            else:
                print("ADDING {0} {1}".format(peak, thar_line_selected))
                self.wavelength_points[self.extraction_index].append((peak, thar_line_selected))

            
            self.update_wavelength_solution()

        else:
            print("Add wavelength point {}".format(event))

            #wavelength = input("Specify the approximate wavelength at pixel {0}:\n"
            #    .format(event))

            self._waiting_for_keypress = True
            print("Waiting for keypress..")
            self._keypress_callback = lambda x: self._add_wavelength_point(event, x)


    def save(self, filename):
        with open(filename, "w") as fp:
            pickle.dump(self.wavelength_points, fp, -1)

        print("saved to {}".format(filename))

    def load(self, filename):
        with open(filename, "r") as fp:
            self.wavelength_points = pickle.load(fp)

        print("loaded from {}".format(filename))
        self.update_wavelength_solution()


    def _add_wavelength_point(self, event, wavelength):

        print("_add_wavelength_point {0} and {1}".format(event, wavelength))

        try:
            wavelength = float(wavelength)
        except:
            print("Wavelength input value {} was not valid".format(wavelength))
            return

        point = (event.xdata, wavelength)
        points = [] + self.wavelength_points[self.extraction_index]
        points.append(point)

        assert self._check_wavelength_points(points)
        self.wavelength_points[self.extraction_index].append(point)

        self.update_wavelength_solution()


    def update_wavelength_solution(self, max_order=3):

        # Update the wavelength solution for the current extraction index
        points = np.array(self.wavelength_points[self.extraction_index])

        if 2 >= points.size:

            if not self.extrapolate_wavelength_solution():
                y = self.extracted_arc_spectra[self.extraction_index]
                x = np.arange(y.size)
                self._x2 = x
                self.axes[1].lines[0].set_data(x, y)
                self.axes[1].set_xlim(x[0], x[-1])
                self.axes[1].set_ylim(0, np.nanmax(y))
                self.figure.canvas.draw()

                print("Add another point")
                return None
            else:
                return

        points = points[points[:, 0].argsort()]
        
        assert self._check_wavelength_points(points)

        order = np.clip(len(points) - 1, 0, max_order)
        model = np.polyfit(points[:, 0], points[:, 1], order)

        # Find any lines within this
        y = self.extracted_arc_spectra[self.extraction_index]
        x = np.polyval(model, np.arange(y.size))

        # Get any ThAr lines in this region
        thar_lines_to_show = (self.thar_lines >= x.min()) * (x.max() >= self.thar_lines)

        print("Thar lines in region {0} to {1}:".format(x.min(), x.max()))
        print(self.thar_lines[thar_lines_to_show])

        ylim = self.axes[1].get_ylim()
        height = 0.95 * np.ptp(ylim) + ylim[0]
        offsets = np.vstack([
            self.thar_lines[thar_lines_to_show],
            np.ones(thar_lines_to_show.sum()) * height
        ]).T
        self._plotted_arc_lines.set_offsets(offsets)

        self._x2 = x
        self.axes[1].lines[0].set_data(x, y)
        self.axes[1].set_xlim(x[0], x[-1])

        self.figure.canvas.draw()


    def extrapolate_wavelength_solution(self):
        return False

        # Get nearest point with some measurements
        wl_points = np.array([np.array(e).size for e in self.wavelength_points])

        if not np.any(wl_points > 2):
            return False

        indices = np.where(wl_points > 2)[0]
        differences = self.extraction_index - indices
        min_difference = np.argsort(np.abs(differences))

        points = np.array(self.wavelength_points[indices[min_difference]])
        points = points[points[:,0].argsort()]

        order = np.clip(len(points) - 1, 0, max_order)
        model = np.polyfit(points[:,0], points[:,1], order)
        y = self.extracted_arc_spectra[self.extraction_index].size
        pixel_size = np.median(np.diff(np.polyval(model, np.arange(y.size))))

        scale = np.arange(y.size)
        if model[0] < model[-1]:
            scale = -scale
        model = pixel_size * scale + differences[min_difference] * y.size

        # Get any ThAr lines in this region
        thar_lines_to_show = (self.thar_lines >= x.min()) * (x.max() >= self.thar_lines)

        print("Thar lines in region {0} to {1}:".format(x.min(), x.max()))
        print(self.thar_lines[thar_lines_to_show])

        ylim = self.axes[1].get_ylim()
        height = 0.95 * np.ptp(ylim) + ylim[0]
        offsets = np.vstack([
            self.thar_lines[thar_lines_to_show],
            np.ones(thar_lines_to_show.sum()) * height
        ]).T
        self._plotted_arc_lines.set_offsets(offsets)

        self._x2 = x
        self.axes[1].lines[0].set_data(x, y)
        self.axes[1].set_xlim(x[0], x[-1])

        self.figure.canvas.draw()






    def _check_wavelength_points(self, points):

        return True

        points = np.atleast_2d(points)
        if points.size == 0:
            # No points. This is OK.
            return True

        points = points[points[:, 0].argsort()]
        sign_difference = np.sign(np.diff(points[:, 1]))

        if np.all(sign_difference > 0) or np.all(sign_difference < 0):
            # things look OK.
            return True

        else:
            print("Difference between the sign points is not the same direction"
                ": {}".format(np.diff(points[:, 1])))
            return False






def identify_emission_lines(data, num=5, **kwargs):
    """
    Identify the `num` largest emission peaks in the data.

    """

    data = data.flatten()
    finite = np.isfinite(data)

    sigma_detect_threshold = kwargs.pop("sigma_detect_threshold", 3)

    while True:

        sigma = (data - np.median(data[finite]))/np.std(data[finite])
        indices = np.where(sigma > sigma_detect_threshold)[0]

        # Group neighbouring pixels and find the pixel mid-point of those groups
        groups = np.array_split(indices, np.where(np.diff(indices) != 1)[0] + 1)
        means = np.array([np.argmax(data[g]) + g[0] for g in groups])

        if len(means) == num:
            break

        elif len(means) > num:
            means = means[np.argsort(data[means])[::-1][:num]]
            break

        sigma_detect_threshold *= 0.9
        if sigma_detect_threshold < 0.1:
            raise RuntimeError("could not find {} peaks".format(num))

    #peaks = data[peaks]
    #return (means, peaks)
    return means

    """
    # Refine on the means.
    profiles = []
    stddev = kwargs.pop("stddev", 1.)
    neighbours = kwargs.pop("neighbours", 10)
    for mean in means:

        profile = EmissionProfile()
        indices = slice(mean - neighbours, mean + neighbours + 1)

        x = np.arange(data.size)[indices]
        y = data[indices]

        default_p0 = dict(b=min(y), amplitude=data[mean], stddev=stddev,
            mean=mean)
        for k, v in default_p0.items():
            setattr(profile, k, v)

        fit = fitting.LevMarLSQFitter()
        profiles.append(fit(profile, x, y))

    return profiles
    """


def find_wavelength_solutions(means, thar_lines, degree=1, full_output=False):

    # Sort the means from left to right
    means = np.sort(means)
    distances = means[1:] - means[0]

    # Calculate distances between all peaks

    # Go through every peak in the line list. Allowing for a free parameter in
    # scaling, calculate the sum of the distance.
    thar_lines = np.sort(thar_lines)
    euclidian_distance = np.zeros(thar_lines.size - means.size)
    for i, line in enumerate(thar_lines[:-means.size]):

        thar_distances = thar_lines[i + 1:i + means.size] - line
        assert thar_distances.size == distances.size

        coefficients = np.polyfit(distances, thar_distances, degree)
        model = np.polyval(coefficients, distances)

        raise a
        euclidian_distance[i] = ((model - thar_distances)**2).sum()

    if full_output:
        return (thar_lines[euclidian_distance.argmin()], euclidian_distance)
    return thar_lines[euclidian_distance.argmin()]


class ArcFrame(CCD):

    pass
