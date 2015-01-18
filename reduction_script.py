# coding: utf-8

""" Reduction script for APF data. """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

# Standard library
import logging
from glob import glob

# Third-party
import numpy as np

# Module-specific
import ccd
import utils
from science import ScienceFrame
from observing import ObservingSequence

# Create logger.
logger = logging.getLogger(__name__)


# Let's get an overview of all the data.
observed_frames = ObservingSequence(glob("data/apfeng?????.fits"))

"""
# Find groups of sequential flat fields.
_ = np.where(observed_frames["OBJECT"] == "WideFlat")[0]
sequential_flat_fields = np.array_split(_, np.where(np.diff(_) != 1)[0] + 1)

all_flat_images = []
master_flat_filenames = []
for i, group_of_sequential_flat_fields in enumerate(sequential_flat_fields):

    # Load the flat field frames and overscan-correct them.
    images = [ccd.CCD.from_filename(image["FILENAME"]).subtract_overscan() \
        for image in observed_frames[group_of_sequential_flat_fields]]
    all_flat_images.extend(images)

    # Combine this set of flat fields.
    group_master_flat = ccd.combine_data(images, method="median")

    # Save the master flat with an appropriate name.
    image_limits = utils.parse_image_limits_from_sequence(
        observed_frames[group_of_sequential_flat_fields]["FILENAME"])
    filename = "median-flat-{0}-{1}.fits".format(*image_limits)
    master_flat_filenames.append(filename)
    group_master_flat.writeto(filename)

    # Create a master flat of all flat fields?
    if i + 1 == len(sequential_flat_fields):
        filename = "median-flat-all.fits"
        master_flat = ccd.combine_data(all_flat_images, method="median")
        master_flat.writeto(filename)
        master_flat_filenames.append(filename)
        del images, all_flat_images

"""
master_flat = ccd.CCD.from_filename("median-flat-all.fits", live_dangerously=True)

# Select some reasonably high-S/N image to perform the trace.
high_snr_star = ScienceFrame.from_filename("data/apfeng10070.fits")
apertures = high_snr_star.fit_apertures()
aperture_trace_coefficients = high_snr_star.trace_apertures(apertures)

foo = high_snr_star.extract_aperture(apertures[0], aperture_trace_coefficients[0],
    live_dangerously=True)

# Normalise the master flat field(s).
master_flat.data /= master_flat.imstat["median"]
master_flat.writeto("normalised-median-all-flats.fits", clobber=True)

non_science_objects = ("Dark", "Iodine", "NarrowFlat", "ThAr", "WideFlat")
science_indices = np.where([f not in non_science_objects \
    for f in observed_frames["OBJECT"]])[0]

for row in observed_frames[science_indices]:

    # Load the science frame and subtract the overscan
    science_image = ccd.CCD.from_filename(row["FILENAME"])
    science_image = science_image.subtract_overscan()
    science_image.writeto(row["FILENAME"].replace(".fits", "-oc.fits"), clobber=True)

    # Clean the science frames of cosmic rays.

    # Divide the normalised flat field into the science images.
    science_image.data /= master_flat.data

    # Trace the other science frames?
    science_image.writeto(row["FILENAME"].replace(".fits", "-check.fits"), clobber=True)
    raise a

    # Extract the science spectra
    extracted_spectra = science_image.extract_apertures(apertures,
        aperture_trace_coefficients)



# Overscan-correct the ThAr frames.
# Combine the ThArs?

# Extract the ThAr lamps

# Identify the ThAr lines and map pixels to wavelength

# Apply wavelength calibration to science frames.

# Iodine cell?
