        # Do the aperture widths change across the CCD?
        trace_image = pipeline.ScienceFrame.from_filename(os.path.join(
            REDUCED_DATA_DIR, "NarrowFlat-median-all.fits"))
        trace_image.subtract_overscan()


        N = 10 # draws
        indices = np.linspace(0, trace_image.shape[0], N + 2).astype(int)[1:-1]

        all_apertures = []
        for index in indices:
            apertures = trace_image.fit_apertures(index)
            all_apertures.append(apertures)


        fig, ax = plt.subplots()
        for index, apertures in zip(indices, all_apertures):

            means = [aperture.mean.value for aperture in apertures]
            widths = [aperture.width.value for aperture in apertures]

            scat = ax.scatter(means, widths, cmap=matplotlib.cm.copper, c=index * np.ones(len(means)), vmin=0, vmax=trace_image.shape[0])

        cbar = plt.colorbar(scat)
        cbar.set_label(r"$x$ $[{\rm pixels}]$")
        ax.set_xlabel(r"$y$ $[{\rm pixels}]$")
        ax.set_ylabel(r"$\sigma$ $[{\rm pixels}]$")
        ax.set_xticks([0, trace_image.shape[1]/4, trace_image.shape[1]/2,
            trace_image.shape[1] * 3 / 4., trace_image.shape[1]])
        ax.yaxis.set_major_locator(MaxNLocator(4))

        ax.set_xlim(0, trace_image.shape[1])
        ax.set_ylim(0, 4)


        fig.tight_layout()

        fig.savefig("aperture-width-across-ccd.eps", dpi=300)
        fig.savefig("aperture-width-across-ccd.pdf", dpi=300)
        fig.savefig("aperture-width-across-ccd.png", dpi=300)
