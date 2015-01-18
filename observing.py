# coding: utf-8

""" Classes for dealing with a sequence of data frames. """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

# Standard library
import logging

# Third-party
from astropy.io import fits
from astropy.table import Table

# Create logger
logger = logging.getLogger(__name__)


_observing_sequence_header_keywords = ["RA", "DEC", "HA", "OBJECT", 
    "OBSTYPE", "DATE-OBS", "EXPTIME", "DATASEC", "PPRERD", "ROVER", "COVER"]
def ObservingSequence(filenames, additional_keywords=None, null_value=None):
    """
    Create an observing table with relevant header information for a general
    overview of all the data available.

    :param filenames:
        Filenames of sequentially observed data frames in a given night.

    :type filenames:
        list of str

    :param additional_keywords: [optional]
        Additional header keywords to include in the resulting table. By default
        the header keywords included are: %s

    :type additional_keywords:
        list of str

    :param null_value: [optional]
        The value to use when a header keyword is not present in a given file.
    """

    header_keywords = [] + _observing_sequence_header_keywords
    if additional_keywords is not None:
        if not isinstance(additional_keywords, (tuple, list)):
            raise TypeError("additional_keywords should be a list or tuple")
        header_keywords.extend(additional_keywords)

    data = []
    for filename in filenames:
        header = fits.getheader(filename)
        data.append([filename] \
            + [header.get(k, null_value) for k in header_keywords])

    table = Table(map(list, zip(*data)), names=["FILENAME"] + header_keywords,
        meta={"null_value": null_value})
    table.sort("DATE-OBS")
    return table

ObservingSequence.__doc__ %= ", ".join(_observing_sequence_header_keywords)