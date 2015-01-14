# coding: utf-8

""" Class for dealing with science frames. """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

# Standard library
import logging

# Third-party
import numpy as np

from ccd import CCD

# Create logger
logger = logging.getLogger(__name__)


class ScienceFrame(CCD):

    def trace(self):
        # trace orders
        raise NotImplementedError


    def extract(self, wavelength_calibration=None):
        raise NotImplementedError

