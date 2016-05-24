"""
Examples to help illustrate the module BPL.PY
=============================================

Created: Mon May 23, 2016  02:56PM
Last modified: Tue May 24, 2016  06:36PM
Copyright: Bedartha Goswami <goswami@uni-potsdam.de>

"""

import numpy as np
import matplotlib.pyplot as pl
import bpl


def show_examples():
    """
    Returns precipitation time series with power-law-like event distribution.
    """
    # set power-law parameters
    alpha = 2.5
    xmin = 1E1
    xmax = 1E5
    size = 1E5

    # get random samples from the bounded power-laws
    s1 = bpl.sample(alpha=alpha, size=size, xmin=xmin, xmax=None)
    s2 = bpl.sample(alpha=alpha, size=size, xmin=xmin, xmax=xmax)
    t = np.arange(size)

    # choose a random window of 1000 points
    start = np.random.randint(low=0, high=size, size=1)
    stop = start + 1000

    # plot time series with different markers for extreme & anomalous events
    fig = pl.figure(figsize=[14.5, 6.5])
    # power-law time series bounded only from below
    ax1 = fig.add_axes([0.07, 0.57, 0.35, 0.35])
    ax1.fill_between(t[start:stop], s1[start:stop],
                     facecolor="RoyalBlue", edgecolor="none")
    ax1.set_title("Power-law random variate snapshot\n(bounded only below)",
                  fontsize=12)
    # power-law time series bounded from above and below
    ax2 = fig.add_axes([0.07, 0.07, 0.35, 0.35])
    ax2.fill_between(t[start:stop], s2[start:stop],
                     facecolor="IndianRed", edgecolor="none")
    ax2.set_title("Power-law random variate snapshot\n(bounded above & below)",
                  fontsize=12)
    ax2.set_xlabel("Time (units)", fontsize=12)
    for ax in fig.axes:
        ax.set_ylabel("Observable (units)", fontsize=12)
        ax.set_xlim(start, stop)

    # plot histograms and PDFs
    ax3 = fig.add_axes([0.51, 0.09, 0.35, 0.85])
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    # power-law time series bounded only from below
    h1, be1 = bpl.histogram(s1, bins=None, density=True,
                            ax=None, plot=False)
    mp1 = 0.5 * (be1[:-1] + be1[1:])
    ax3.plot(mp1, h1, "o", mec="RoyalBlue", mfc="none", ms=8, mew=1.1)
    ylo, yhi = ax3.get_ylim()
    xlo, xhi = ax3.get_xlim()
    x1 = np.logspace(np.log10(xlo), np.log10(xhi),
                     num=100, base=10)
    pl1 = bpl.pdf(x1, alpha, xmin, xmax=None)
    ax3.plot(x1, pl1, "-", c="RoyalBlue", lw=1.5, zorder=-1)
    # power-law time series bounded from above and below
    h2, be2 = bpl.histogram(s2, bins=None, density=True,
                            ax=None, plot=False)
    mp2 = 0.5 * (be2[:-1] + be2[1:])
    ax3.plot(mp2, h2, "s", mec="IndianRed", mfc="none", ms=8, mew=1.1)
    x2 = np.logspace(np.log10(xmin), np.log10(xmax),
                     num=100, base=10)
    pl2 = bpl.pdf(x2, alpha, xmin, xmax=xmax)
    ax3.plot(x2, pl2, "-", c="IndianRed", lw=1.5, zorder=-1)
    # a few prettification adjustments
    ax3.set_ylim(ylo, yhi)
    ax3.grid(which="both", axis="both")
    ax3.legend(["Histogram (bounded below)",
                "Theoretical (bounded below)",
                "Histogram (bounded both sides)",
                "Theoretical (bounded both sides)"
                ])
    ax3.set_xlabel("Observable (units)", fontsize=12)
    ax3.set_ylabel("Probability", fontsize=12)

    # save/show plot
    pl.show(fig)
    return None

if __name__ == "__main__":
    print("running example...")
    show_examples()
    print("done.")
