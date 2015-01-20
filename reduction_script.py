# coding: utf-8

""" Reduction script for APF data. """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

# Standard library
import cPickle as pickle
import logging
import os
from glob import glob

# Third-party
import numpy as np
import pipeline

# from astropy import specutils
from oracle import specutils

# Initialise logging
logger = logging.getLogger("pipeline")

# Let's get an overview of all the data.
observed_frames = pipeline.observing.sequence(glob("raw_data/apfeng?????.fits"),
    additional_keywords=["ICELNAM", "OMEGAPOW"])

# Options
clobber = True
COMBINE_FLATS = False

TRACE_APERTURES = True
TRACE_FILENAME = "apfeng10070.fits"

SOLVE_WAVELENGTHS = False

EXTRACT_SPECTRA = True

# Flat fields
flat_field_combination_method = "median"
if COMBINE_FLATS:
    # Find groups of sequential flat fields.
    _ = np.where(observed_frames["OBJECT"] == "WideFlat")[0]
    sequential_flat_fields = np.array_split(_, np.where(np.diff(_) != 1)[0] + 1)

    all_flat_images = []
    master_flat_filenames = []
    for i, group_of_sequential_flat_fields in enumerate(sequential_flat_fields):

        # Load the flat field frames and overscan-correct them.
        images = [pipeline.CCD.from_filename(image["FILENAME"]).subtract_overscan() \
            for image in observed_frames[group_of_sequential_flat_fields]]
        all_flat_images.extend(images)

        # Combine this set of flat fields.
        group_master_flat = pipeline.ccd.combine_data(images,
            method=flat_field_combination_method)

        # Save the master flat with an appropriate name.
        image_limits = utils.parse_image_limits_from_sequence(
            observed_frames[group_of_sequential_flat_fields]["FILENAME"])
        filename = "reduced_data/{0}-flat-{0}-{1}.fits".format(
            flat_field_combination_method, *image_limits)
        master_flat_filenames.append(filename)
        logger.info("Saved combined flat to {}".format(filename))
        group_master_flat.writeto(filename, clobber=clobber)

        # Create a master flat of all flat fields?
        if i + 1 == len(sequential_flat_fields):
            filename = "reduced_data/{}-flat-all.fits".format(
                flat_field_combination_method)
            master_flat = pipeline.ccd.combine_data(all_flat_images,
                method=flat_field_combination_method)
            master_flat.writeto(filename, clobber=clobber)
            logger.info("Saved master flat to {}".format(filename))
            master_flat_filenames.append(filename)
            del images, all_flat_images

    # Normalise the master flat field(s).
    master_flat.data /= master_flat.imstat[flat_field_combination_method]
    filename = "reduced_data/normalised-{}-flat-all.fits".format(
        flat_field_combination_method)
    master_flat.writeto(filename, clobber=clobber)
    logger.info("Saved normalised master flat to {}".format(filename))

else:
    filename = "reduced_data/normalised-{}-flat-all.fits".format(
        flat_field_combination_method)
    logger.info("Loading normalised master flat from {}".format(filename))
    master_flat = pipeline.CCD.from_filename(filename)


if TRACE_APERTURES:
    logger.info("Tracing apertures..")
    high_snr_star = pipeline.ScienceFrame.from_filename("raw_data/{}".format(
        TRACE_FILENAME))
    apertures = high_snr_star.fit_apertures()
    coefficients = high_snr_star.trace_apertures(apertures)

    with open("reduced_data/apertures_coefficients.pickle", "w") as fp:
        pickle.dump((apertures, coefficients), fp, -1)

else:
    logger.info("Loading apertures from existing file")
    with open("reduced_data/apertures_coefficients.pickle", "r") as fp:
        apertures, coefficients = pickle.load(fp)


if SOLVE_WAVELENGTHS:
    raise NotImplementedError("Nobody wants to do this.")

else:
    with open("reduced_data/wavelength_mapping.pickle", "r") as fp:
        wavelength_mapping = pickle.load(fp)


if EXTRACT_SPECTRA:
    extract_apertures = [20, 21, 22, 23, 40]
    non_science_objects = ("Dark", "Iodine", "NarrowFlat", "ThAr", "WideFlat")
    science_indices = np.where([f not in non_science_objects \
        for f in observed_frames["OBJECT"]])[0]

    median_aperture_stddev = np.median([_.stddev for _ in apertures])
    for row in observed_frames[science_indices]:

        # Load the science frame and subtract the overscan
        science_image = pipeline.ScienceFrame.from_filename(row["FILENAME"])
        science_image = science_image.subtract_overscan()

        # Clean the science frames of cosmic rays.
        science_image = science_image.clean_cosmic_rays()

        # Divide the normalised flat field into the science images.
        science_image.data /= master_flat.data

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
            spectrum.save("reduced_data/{0}_es_{1:.0f}{2}".format(basename, index,
                ext), clobber=clobber)

    else:
        print("No spectra extracted. Nothing left to do.")



