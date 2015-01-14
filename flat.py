# coding: utf-8

""" Class for dealing with flat field frames. """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

# Standard library
import logging

# Third-party
import numpy as np

from ccd import CCD

# Create logger
logger = logging.getLogger(__name__)


class FlatFieldFrame(CCD):

    def normalise(self, trace_info):
        pass
