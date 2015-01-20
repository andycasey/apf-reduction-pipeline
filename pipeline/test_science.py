
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

    for aperture in apertures:
        ax.axvline(aperture.mean, c='b')

        # Get some data around here
        #i = int(midpoint)
        #i = (i - int(aperture_width), i + int(aperture_width) + 1)
        #x = np.arange(*i)
        x = np.arange(aperture.mean - 10, aperture.mean + 10, 0.01)
        ax.plot(x, aperture(x), c='m')


    #for midpoint in new_midpoints:
    #    ax.axvline(midpoint, c='r')

    ax.set_xlim(0, data_slice.size)
    ax.set_ylim(0, data_slice.max())

    plt.show()


    #obs.trace_apertures()
    #aperture = apertures[17]
    #coefficients = obs.trace_aperture(aperture)

    """
    x, offsets, coefficients = obs.trace_apertures_approximately(apertures)



    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
    ax[0].imshow(obs.data, cmap=plt.cm.Greys_r, vmin=0, vmax=1500, aspect="auto")
    ax = ax[1]
    ax.imshow(obs.data, cmap=plt.cm.Greys_r, vmin=0, vmax=1500, aspect="auto")
    ax.set_color_cycle(["r", "b"])

    y = np.arange(obs.data.shape[0])
    ax.plot(x, y, c="b")
    for offset in offsets:
        ax.plot(x - offset, y, c="r")
    
    ax.set_xlim(-0.5, obs.data.shape[1] + 0.5)
    ax.set_ylim(-0.5, obs.data.shape[0] + 0.5)

    plt.show()

    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
    ax[0].imshow(obs.data, cmap=plt.cm.Greys_r, vmin=0, vmax=1500, aspect="auto")
    ax = ax[1]
    ax.imshow(obs.data, cmap=plt.cm.Greys_r, vmin=0, vmax=1500, aspect="auto")
    ax.set_color_cycle(["r", "b"])

    coefficients, outliers, corrected = obs.trace_apertures(apertures)
    y = np.arange(obs.data.shape[0])
    for c, o, p in zip(coefficients, outliers, corrected):

        color = "g" if p else ["b", "r"][o]
        x = np.polyval(c, y)
        ax.plot(x, y, c=color)
    
    ax.set_xlim(-0.5, obs.data.shape[1] + 0.5)
    ax.set_ylim(-0.5, obs.data.shape[0] + 0.5)

    plt.show()


