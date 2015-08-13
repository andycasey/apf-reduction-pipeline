# coding: utf-8

""" Classes for dealing with a sequence of data frames. """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

# Standard library
import logging
import time

# Third-party
import numpy as np
from astropy.io import fits
from astropy.table import Table
from . import (ccd, )

# Create logger
logger = logging.getLogger(__name__)


_observing_sequence_header_keywords = ["RA", "DEC", "HA", "OBJECT", "OBSTYPE",
    "DATE-OBS", "EXPTIME", "DATASEC", "PPRERD", "ROVER", "COVER", "DECKRNAM"]


class Sequence(object):

    def __init__(self, filenames, additional_keywords=None, null_value=None):
        """
        Create an observing table with relevant header information for a general
        overview of all the data available.

        :param filenames:
            Filenames of sequentially observed data frames in a given night.

        :type filenames:
            list of str

        :param additional_keywords: [optional]
            Additional header keywords to include in the resulting table. By
            default the header keywords included are: %s

        :type additional_keywords:
            list of str

        :param null_value: [optional]
            The value to use when a header keyword is not present in a ile.
        """.format(_observing_sequence_header_keywords)

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

        self.observations = Table(map(list, zip(*data)),
            names=["FILENAME"] + header_keywords,
            meta={"null_value": null_value})
        self.observations.sort("DATE-OBS")

        return None


    def get_nearest_observation(self, image, object_type, same_decker=True):
        """
        Return the nearest observation of a specific type to the observation
        provided.
        """

        match = (self.observations["OBJECT"] == object_type)
        if same_decker:
            match *= (self.observations["DECKRNAM"] == image.meta["DECKRNAM"])

        if not any(match):
            return []

        # self.observations is already sorted by DATE-OBS
        index = np.searchsorted(self.observations["DATE-OBS"][match],
            image.meta["DATE-OBS"])

        if index == match.sum() - 1:
            return index # it's the one just beforehand (also the last entr)

        # Before or after: which is closer?
        i, s = np.where(match)[0], "%Y-%m-%dT%H:%M:%S"
        x = time.mktime(time.strptime(image.meta["DATE-OBS"].split(".")[0], s))
        before, after = [time.mktime(time.strptime(_.split(".")[0], s)) \
            for _ in self.observations["DATE-OBS"][match][index:index+2]]
        return i[index] if abs(before - x) < abs(after - x) else i[index + 1]


    def get_sequential_observations(self, object_type, same_decker=True):
        """
        Return sequential observations in groups.
        """

        match = self.observations["OBJECT"] == object_type
        if same_decker:
            groups = []
            for decker in set(self.observations["DECKRNAM"][match]):
                sub_match = match * (self.observations["DECKRNAM"] == decker)
                _ = np.where(sub_match)[0]
                groups.extend(np.array_split(_, np.where(np.diff(_) != 1)[0]+1))

        else:
            _ = np.where(match)[0]
            groups = np.array_split(_, np.where(np.diff(_) != 1)[0] + 1)
        return groups
 

    def combine_sequential_images(self, object_type, same_decker=True,
        method="median", subtract_overscan=True, clean_cosmic_rays=True,
        **kwargs):

        groups = self.get_sequential_observations(object_type, same_decker)

        combined_images = []
        for group in groups:
            logger.debug("Combining sequential {0} frames:\n{1}".format(
                object_type, self.observations[group]))

            images = []
            for filename in self.observations["FILENAME"][group]:
                image = ccd.CCD.from_filename(filename)
                if subtract_overscan:
                    image.subtract_overscan()
                if clean_cosmic_rays:
                    image.clean_cosmic_rays()
                images.append(image)
            combined_images.append(ccd.combine_data(images, method=method))

        return (combined_images, groups)


    def combine_images(self, object_type, same_decker=True, method="median",
        subtract_overscan=True, clean_cosmic_rays=True, **kwargs):

        match = self.observations["OBJECT"] == object_type
        deckers = set(self.observations["DECKRNAM"][match])

        combined_images = []
        for decker in deckers:
            sub_match = match * (self.observations["DECKRNAM"] == decker)

            images = []
            for filename in self.observations["FILENAME"][sub_match]:
                image = ccd.CCD.from_filename(filename)
                if subtract_overscan:
                    image.subtract_overscan()
                if clean_cosmic_rays:
                    image.clean_cosmic_rays()
                images.append(image)
            combined_images.append(ccd.combine_data(images, method=method))

        return combined_images if same_decker else combined_images[0]


