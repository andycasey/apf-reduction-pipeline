# coding: utf-8

""" Utility functions """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

# Standard library
import os
import re

def parse_image_number(filename):
    return re.sub(r"\D", "", os.path.basename(filename))

def parse_image_limits_from_sequence(filenames):
    numbers = map(parse_image_number, filenames)
    return (min(numbers), max(numbers))