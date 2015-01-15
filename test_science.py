
# coding: utf-8

""" Test the science class. """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"


import science


if __name__ == "__main__":
    
    obs = science.ScienceFrame.from_filename("data/apfeng10088.fits")

    obs.subtract_overscan()
    apertures = obs.fit_apertures(obs.data.shape[0]/2)


    import matplotlib.pyplot as plt

    index = [obs.data.shape[0]/2, None]
    data_slice = obs.data[index].flatten()

    fig, ax = plt.subplots()
    ax.plot(data_slice, 'k')

    for x, aperture in apertures:
        ax.axvline(aperture.mean, c='b')

        # Get some data around here
        #i = int(midpoint)
        #i = (i - int(aperture_width), i + int(aperture_width) + 1)
        #x = np.arange(*i)
        ax.plot(x, aperture(x), c='m')


    #for midpoint in new_midpoints:
    #    ax.axvline(midpoint, c='r')

    ax.set_xlim(0, data_slice.size)
    ax.set_ylim(0, data_slice.max())

    plt.show()


    #obs.trace_apertures()
    aperture = apertures[17][1]
    obs.trace_aperture(aperture)
    raise a