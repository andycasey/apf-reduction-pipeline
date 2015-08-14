# coding: utf-8

""" Reduction script for APF 2015B data. """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

import cPickle as pickle
import logging
import matplotlib
matplotlib.rcParams["text.usetex"] = True
from matplotlib.ticker import MaxNLocator

import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob

import pipeline
from oracle import specutils

logger = logging.getLogger("pipeline")


# CONFIGURATION
RAW_DATA_DIR = "../data/20150801/raw/"
REDUCED_DATA_DIR = "../data/20150801/reduced/"

CLOBBER = True

TRACE_OBJECT = "HR6827"
REDUCTION_STEPS = {
    "COMBINE_FLATS": False,
    "NORMALISE_FLAT": True,
    "TRACE_APERTURES": ["rockosi10027.fits", "rockosi10047.fits"], # HR6827
    "COMBINE_THARS": True,
}

# Optional extras.
COMBINE_FLAT_KWARGS = {}
TRACE_APERTURE_KWARGS = {}

# Now the pain begins.
data = pipeline.observing.Sequence(glob("{}*.fits".format(RAW_DATA_DIR)),
    additional_keywords=["ICELNAM", "OMEGAPOW"])

# Combine flats.
if REDUCTION_STEPS.get("COMBINE_FLATS", True):

    kwds = {
        "method": "median",
        "clean_cosmic_rays": False
    }
    kwds.update(COMBINE_FLAT_KWARGS)
    logger.info("Combining flats using keywords: {}".format(kwds))

    # Combine Wide Flats
    combined_flats, idx = data.combine_sequential_images("WideFlat", **kwds)
    for combined_flat, indices in zip(combined_flats, idx):
        start, end = pipeline.utils.parse_image_limits_from_sequence(
            data.observations["FILENAME"][indices])
        filename = os.path.join(REDUCED_DATA_DIR,
            "WideFlat-{s}-{e}.fits".format(s=start, e=end))
        logger.info("Saving combined WideFlat frame to {}".format(filename))
        combined_flat.writeto(filename, clobber=CLOBBER)

    # Combine sequential Narrow Flats
    combined_flats, idx = data.combine_sequential_images("NarrowFlat", **kwds)
    for combined_flat, indices in zip(combined_flats, idx):
        start, end = pipeline.utils.parse_image_limits_from_sequence(
            data.observations["FILENAME"][indices])
        filename = os.path.join(REDUCED_DATA_DIR,
            "NarrowFlat-{s}-{e}.fits".format(s=start, e=end))
        logger.info("Saving combined NarrowFlat frame to {}".format(filename))
        combined_flat.writeto(filename, clobber=CLOBBER)

    # Combine *all* narrow flats for a good trace.
    _kwds = kwds.copy()
    [_kwds.pop(_, None) for _ in ("same_decker", "clean_cosmic_rays")]
    combined_narrow_flat = data.combine_images("NarrowFlat", same_decker=False,
        clean_cosmic_rays=False, **_kwds)
    filename = os.path.join(REDUCED_DATA_DIR, "NarrowFlat-all.fits")
    logger.info("Saving combined NarrowFlat (all images) to {}".format(filename))
    combined_narrow_flat.writeto(filename, clobber=CLOBBER)

    # Combine *all* wide flats for a normalised flat field.
    _kwds = kwds.copy()
    _kwds.pop("clean_cosmic_rays", None)
    combined_wide_flat = data.combine_images("WideFlat", same_decker=True,
        clean_cosmic_rays=False, **_kwds)[0]
    filename = os.path.join(REDUCED_DATA_DIR, "WideFlat-all.fits")
    logger.info("Saving combined WideFlat (all images) to {}".format(filename))
    combined_wide_flat.writeto(filename, clobber=CLOBBER)

    # Normalise this flat by the mode.
    combined_wide_flat.normalise(method="mode")
    filename = os.path.join(REDUCED_DATA_DIR, "WideFlat-all-normalised.fits")
    logger.info("Saving normalised combined WideFlat (all images) to {}".format(
        filename))
    combined_wide_flat.writeto(filename, clobber=CLOBBER)


# Let's use combined NarrowFlat frames to trace the location of the orders.
if REDUCTION_STEPS.get("TRACE_APERTURES", True):

    kwds = {}
    kwds.update(TRACE_APERTURE_KWARGS)

    trace_filename = os.path.join(REDUCED_DATA_DIR, "NarrowFlat-all.fits")
    image = pipeline.ScienceFrame.from_filename(trace_filename)
    apertures = image.fit_apertures(**kwds)
    coefficients = image.trace_apertures(apertures, **kwds)

    basename = os.path.splitext(trace_filename)[0]

    # Create some figures.
    fig = image.plot_apertures(apertures)
    fig.savefig("{0}-apertures.png".format(basename), dpi=300)

    fig = image.plot_aperture_trace(coefficients)
    fig.savefig("{0}-aperture-trace.png".format(basename), dpi=300)

    filename = "{0}-coefficients.pkl".format(basename)
    logger.info("Saving aperture trace coefficients from {0} to {1}".format(
        trace_filename, filename))

    with open(filename, "wb") as fp:
        pickle.dump((coefficients, ), fp, -1)

    # Get the nearest observation of HR6827 to this one.
    nearest_trace_index = data.get_nearest_observation(image, TRACE_OBJECT,
        same_decker=False)

    # Load the trace image and prepare it for aperture tracing.
    trace_image = pipeline.ScienceFrame.from_filename(
        data.observations["FILENAME"][nearest_trace_index])
    trace_image.subtract_overscan()


    """
    # Noramlise the trace image by the flat field.
    normalised_flat = pipeline.ccd.CCD.from_filename(
        os.path.join(REDUCED_DATA_DIR, "WideFlat-all-normalised.fits"))
    trace_image /= normalised_flat
    trace_image.writeto("test.fits")
    """

    aperture_widths, aperture_width_coefficients \
        = trace_image.fit_aperture_widths(coefficients)

    raise a





    raise a




if SOLVE_WAVELENGTHS:
    raise NotImplementedError("Nobody wants to do this.")

else:
    with open("wavelength_mapping.pickle", "r") as fp:
        wavelength_mapping = pickle.load(fp)


if EXTRACT_SPECTRA:
    extract_apertures = [20, 21, 22, 23, 40]
    non_science_objects = ("Dark", "Iodine", "NarrowFlat", "ThAr", "WideFlat")
    science_indices = np.where([f not in non_science_objects \
        for f in frames["OBJECT"]])[0]

    median_aperture_stddev = np.median(stddevs)
    for row in frames[science_indices]:

        # Load the science frame and subtract the overscan
        science_image = pipeline.ScienceFrame.from_filename(row["FILENAME"])
        science_image = science_image.subtract_overscan()

        # Clean the science frames of cosmic rays.
        science_image = science_image.clean_cosmic_rays()

        # Divide the normalised flat field into the science images.
        science_image._data /= master_flat.data

        # Extract the science spectra that we want.
        for index in extract_apertures:

            # Estimate a width for this aperture.
            flux = science_image.extract_aperture(coefficients[index],
                width=2.5 * median_aperture_stddev)

            # Apply the wavelength calibration.
            f = pipeline.arc.wavelength_calibration(wavelength_mapping[index])
            dispersion = f(np.arange(flux.size))

            # Save as spectrum.
            # TODO: oracle.specutils is kinda dumb. ensure all headers are strs
            k = science_image.meta.keys()
            str_headers = dict(zip(k, [str(science_image.meta[_]) for _ in k]))
            spectrum = specutils.Spectrum1D(disp=dispersion, flux=flux,
                headers=str_headers)

            basename, ext = os.path.splitext(os.path.basename(row["FILENAME"]))
            spectrum.save(REDUCED_DATA_DIR + "/{0}_es_{1:.0f}{2}".format(basename, index,
                ext), clobber=CLOBBER)

            # Make a plot.
            fig, ax = plt.subplots()
            ax.plot(dispersion, flux, c="k")
            ax.set_xlim(dispersion.min(), dispersion.max())
            ax.set_ylim(0, np.median(flux) + 3 * np.std(flux))
            ax.set_xlabel("Wavelength")
            ax.set_ylabel("Flux")
            filename = REDUCED_DATA_DIR + "/{0}_es_{1:.0f}.png".format(basename, index)
            fig.savefig(filename, dpi=150)
            plt.close("all")
            logger.info("Saved image to {}".format(filename))
            
else:
    print("No spectra extracted. Nothing left to do.")
