# coding: utf-8

""" Class for dealing with CCD data. """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

# Standard library
import os
import sys
import logging

# Third-party
import astropy.units as u
import numpy as np
import pyfits
from astropy.nddata import NDData, FlagCollection

# Create logger
logger = logging.getLogger(__name__)


class CCD(NDData):

    @classmethod
    def load(cls, filename, data_index=0, **kwargs):
        """
        Create a CCD instance from the data contained in the given filename.

        :param filename:
            The FITS data filename.

        :type filename:
            str

        :param data_index:
            The image index (starting at zero) that contains the data.

        :type data_index:
            int
        """

        live_dangerously = kwargs.pop("live_dangerously", False)
        expected_obstype = kwargs.pop("expected_obstype", None)
        with pyfits.open(filename) as image:

            header = image[0].header
            obstype = header.get("OBSTYPE", None)
            if expected_obstype is not None \
            and expected_obstype != obstype \
            and not live_dangerously:
                raise TypeError("expected OBSTYPE of {0} but got {1}".format(
                    expected_obstype, obstype))

            # Set the shape, data
            ccd = cls(image[data_index].data)

            # Pass some meta-data from the header
            ccd.meta["_filename"] = filename
            _ = ("RA", "DEC", "OBJECT", "OBSTYPE", "EXPTIME", "DATE-OBS")
            for key in kwargs.pop("header_keys", _):
                ccd.meta[key] = header.get(key, None)

            # Set the flags for data sections and overscan
            # Note that here I just assume that if it's not data, it's overscan.
            ccd.flags = FlagCollection(shape=ccd.shape)
            ccd.flags["data"] = np.zeros(ccd.shape, dtype=bool)
            ccd.flags["overscan"] = np.ones(ccd.shape, dtype=bool)

            # Get data region from DATASEC
            if not "DATASEC" in header:
                logger.debug("No DATASEC header found. Assuming zero overscan.")
                ccd.flags["data"][:] = True
                ccd.flags["overscan"][:] = False

            else:
                # Get the data shape
                data_indices = _parse_iraf_style_section(header["DATASEC"])
                ccd.meta["_data_shape"] = tuple(map(np.ptp, data_indices))

                data_indices = [slice(*_) for _ in data_indices]
                ccd.flags["data"][data_indices] = True
                ccd.flags["overscan"][data_indices] = False

                # Get the overscan shape
                overscan_rows = header.get("ROVER", 0)
                overscan_columns = header.get("COVER", 0)
                ccd.meta["_overscan_shape"] = _parse_overscan_shape(
                    overscan_rows, overscan_columns)

                flagged_overscan_pixels = (ccd.flags["overscan"] == True).sum()
                expected_overscan_pixels = np.multiply(ccd.shape,
                    (overscan_columns, overscan_rows)).sum()

                if expected_overscan_pixels != flagged_overscan_pixels \
                and not live_dangerously:
                    raise ValueError("expected {0} overscan pixels but flagged "
                        "{1} pixels".format(expected_overscan_pixels,
                            flagged_overscan_pixels))

        return ccd
        

    def subtract_overscan(self):
        """
        Subtract the median of any overscan region in the CCD, and return just
        the overscan-corrected data region.
        """

        if not np.any(self.flags["overscan"]):
            # No overscan; perhaps it's already been subtracted?
            return self

        # We'll need these.
        data_shape = self.meta["_data_shape"]
        overscan_shape = self.meta["_overscan_shape"]

        # Slicing complex Flags is currently not implemented in NDData, so we
        # will have to slice on the .data attribute:
        overscan = self.data[self.flags["overscan"]].reshape(overscan_shape)

        # Make the overscan correction
        self.data = self.data[self.flags["data"]].reshape(data_shape) \
            - np.median(overscan, axis=1)[:,np.newaxis]

        # Update the flags
        self.flags.shape = data_shape
        self.flags["data"] = np.ones(data_shape, dtype=bool)
        self.flags["overscan"] = np.zeros(data_shape, dtype=bool)

        # Update the metadata
        self.meta["_overscan_shape"] = (0, 0)
        self.meta["reduction_log"] = "Overscan corrected."

        return self


def _parse_overscan_shape(rows, columns):
    """
    Parse the number of overscan rows and columns into indices that can be used
    to reshape arrays.

    :param rows:
        The number of overscan rows.

    :type rows:
        int

    :param columns:
        The number of overscan columns.

    :type columns:
        int
    """

    if rows == 0 and columns == 0:
        return (0, 0)

    if rows == 0 and columns > 0:
        return (-1, columns)

    if rows > 0 and columns == 0:
        return (rows, -1)

    if rows > 0 and columns > 0:
        return (rows, columns)


def _parse_iraf_style_section(header_string):
    """
    Parse IRAF/NOAO-style data sections to Python indices.

    :param header_string:
        The IRAF/NOAO-style data section string (e.g., [1:2048,1:4608]).

    :type header_string:
        str
    """

    indices = []
    dimensions = header_string.strip("[] ").split(",")
    for dimension in dimensions:
        start_pixel, end_pixel = map(int, dimension.split(":"))

        # These pixels are inclusively marked.
        start_index, end_index = start_pixel - 1, end_pixel
        indices.append([start_index, end_index])

    # IRAF/NOAO give the image shape the wrong way around
    return indices[::-1]


