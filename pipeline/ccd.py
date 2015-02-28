# coding: utf-8

""" Class for dealing with CCD data. """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

# Standard library
import logging
import os
import sys
from collections import OrderedDict

# Third-party
import astropy.units as u
import cosmics
import numpy as np
from astropy.io import fits
from astropy.nddata import NDData, FlagCollection
from scipy.stats import mode

# Create logger
logger = logging.getLogger(__name__)


class CCD(NDData):

    def __init__(self, data, **kwargs):

        super(CCD, self).__init__(data, **kwargs)

        # Assume it's all data and set overscan/data flags appropriately.
        self.flags = FlagCollection(shape=data.shape)
        self.flags["data"] = np.ones(data.shape, dtype=bool)
        self.flags["overscan"] = np.zeros(data.shape, dtype=bool)

        # Set metadata that we'll need.
        self.meta["_filename"] = None
        self.meta["_data_shape"] = data.shape
        self.meta["_overscan_shape"] = (0, 0)

        self.__header_keys = ("RA", "DEC", "OBJECT", "OBSTYPE", "EXPTIME",
            "DATE-OBS", "DATASEC", "ROVER", "COVER")

    @property
    def imstat(self):
        """
        Return statistical properties about the data sections in the CCD.
        """

        data = self.data[self.flags["data"]]
        return OrderedDict([
            ("pixels", data.size),
            ("non_finite_pixels", (~np.isfinite(data)).sum()),
            ("mean", np.mean(data)),
            ("median", np.median(data)),
            ("stddev", np.std(data)),
            ("min", np.min(data)),
            ("max", np.max(data)),
        ])


    @classmethod
    def from_filename(cls, filename, data_index=0, **kwargs):
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
        with fits.open(filename) as image:

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
            for key in kwargs.pop("header_keys", ccd.__header_keys):
                ccd.meta[key] = header.get(key, None)

            # Set the flags for data sections and overscan
            # Note that here I just assume that if it's not data, it's overscan.
            ccd.flags = FlagCollection(shape=ccd.data.shape)
            ccd.flags["data"] = np.zeros(ccd.data.shape, dtype=bool)
            ccd.flags["overscan"] = np.ones(ccd.data.shape, dtype=bool)

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
                expected_overscan_pixels = np.multiply(ccd.data.shape,
                    (overscan_columns, overscan_rows)).sum()

                if expected_overscan_pixels != flagged_overscan_pixels \
                and not live_dangerously:
                    raise ValueError("expected {0} overscan pixels but flagged "
                        "{1} pixels".format(expected_overscan_pixels,
                            flagged_overscan_pixels))

        return ccd
        

    def writeto(self, filename, clobber=False):
        """
        Write the CCD frame to a filename.

        :param filename:
            The output filename.

        :type filename:
            str

        :param clobber: [optional]
            Overwrite the filename if it already exists.

        :type clobber:
            bool
        """

        if os.path.exists(filename) and not clobber:
            raise ValueError("file exists and we were told not to clobber it")

        header = fits.Header()
        header.update(self.meta)

        hdu = fits.PrimaryHDU(self.data, header=header)
        hdu_list = fits.HDUList([hdu])
        hdu_list.writeto(filename, clobber=clobber)


    def clean_cosmic_rays(self, gain=1, sigclip=8.0, sigfrac=0.5, objlim=5.0,
        **kwargs):

        maxiter = kwargs.pop("maxiter", 2)
        full_output = kwargs.pop("full_output", False)
        image = cosmics.cosmicsimage(self.data, gain=gain, sigclip=sigclip,
            sigfrac=sigfrac, objlim=objlim, **kwargs)
        result = image.run(maxiter=maxiter)

        self._data = image.cleanarray

        if full_output:
            return (self, image)
        return self


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
        self._data = self.data[self.flags["data"]].reshape(data_shape) \
            - np.median(overscan, axis=1)[:,np.newaxis]

        # Update the flags
        self.flags.shape = data_shape
        self.flags["data"] = np.ones(data_shape, dtype=bool)
        self.flags["overscan"] = np.zeros(data_shape, dtype=bool)

        # Update the metadata
        self.meta["_overscan_shape"] = (0, 0)
        self.meta["ROVER"], self.meta["COVER"] = 0, 0
        self.meta["reduction_log"] = "Overscan corrected."

        return self


def combine_data(frames, method="median", **kwargs):
    """
    Combine the data sections in multiple CCD frames. Note that this method will
    discard overscan regions, and it is possible that not all meta-data keys
    will be merged correctly.

    :param frames:
        The list of CCD frames with data sections to combine.

    :type frames:
        list

    :param method: [optional]
        The combination method to use. Available methods are median (default),
        average, or sum.

    :type method:
        str

    :returns:
        A single CCD frame with a combined data section. 
    """

    method = method.lower()
    if method not in ("median", "average", "sum"):
        raise ValueError("method must be either median, average, or sum")

    # Check the frames are the same data shape
    shapes = list(set(_.meta["_data_shape"] for _ in frames))
    if len(shapes) > 1:
        raise ValueError("frames have mis-matching data shapes: {0}".format(
            " and ".join(map(str, shapes))))

    # [TODO] Check they are the same type?

    # Combine the data
    stack_shape = (len(frames), ) + shapes[0]
    data = np.zeros(stack_shape)
    for i, f in enumerate(frames):
        data[i][:] = f.data[f.flags["data"]].reshape(f.meta["_data_shape"])

    func = {
        "median": np.median,
        "mean": np.mean,
        "sum": np.sum
    }[method]
    combined_data = CCD(func(data, axis=0))
    combined_data._meta = frames[0].meta.copy()
    combined_data.meta["_filename"] = "Combined from existing files."
    #combined_data.meta["_filename"] = "Combined [{0}] from {1}".format(method,
    #    "|".join(map(str, [_.meta.get("_filename", None) for _ in frames])))

    return combined_data


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


