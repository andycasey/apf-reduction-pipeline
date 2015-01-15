# coding: utf-8

""" Classes for dealing with a sequence of data frames. """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

# Standard library
import collections
import logging

# Third-party
import numpy as np
import pyfits

from ccd import CCD
from arc import ArcFrame
from flat import FlatFieldFrame
from science import ScienceFrame

# Create logger
logger = logging.getLogger(__name__)


class ObservingSequence(collections.MutableSequence):
    """
    Perhaps this should just read in the files and produce a pandas table/similar
    with the relevant information that we will need.

    Well, we need some way other than loading all the data at once...
    """

    _object_key = "OBJECT"
    _object_translator = {
        "ThAr": ArcFrame,
        "WideFlat": FlatFieldFrame,
    }

    def __init__(self, *args):
        self.list = list()
        self.extend(list(args))
        
    def parse(self, item):
        """
        Load a filename or add a CCD frame.
        """

        if isinstance(item, (str, unicode)):
            # Check the header here and translate to the correct frame.
            with pyfits.open(item) as image:
                obstype = image[0].header.get(self._object_key, None)

            # If we don't recognise the header, default to CCD
            loader_class = self._object_translator.get(obstype, CCD)
            frame = (item, loader_class)

        else:
            raise TypeError("item is not a filename")
        return frame

    def __len__(self):
        return len(self.list)

    def __getitem__(self, i):
        return self.list[i]

    def __delitem__(self, i):
        del self.list[i]

    def __setitem__(self, i, v):
        self.list[i] = self.parse(v)

    def insert(self, i, v):
        self.list.insert(i, self.parse(v))

    def __str__(self):
        return str(self.list)


