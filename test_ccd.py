# coding: utf-8

""" Unit tests for the CCD functionality. """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

from glob import glob

from . import ccd

# Create logger
logger = logging.getLogger(__name__)


def test_loader():
    first_image = ccd.CCD.load(glob.glob("data/*.fits")[0])