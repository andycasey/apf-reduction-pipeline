
# coding: utf-8

""" Test the science class. """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"


import science


if __name__ == "__main__":
    
    obs = science.ScienceFrame.from_filename("data/apfeng10088.fits")

    obs.subtract_overscan()
    obs.trace_apertures()