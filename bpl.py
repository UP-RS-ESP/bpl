"""
Python module to work with bounded power-law (BPL) distributed random variables.

This module contains a few basic functions that help analyse power-law
distributed random variables that can be either bounded only from below or
from both above and below. It provides the following functions:

1.  Generate random samples from power-law distributed random variables.
2.  Estimate a logarithimically binned histogram for a power-law
    distributed random variable.
3.  Estimate the probability density function and the cumulative
    distribution function for specified power-law exponent.

Introduction to BPL
-------------------

The probability density function (PDF) of a power-law variable is at best
bounded from below to avoid divergence of the density near zero. In this case,
the power-law distribution is defined on the support
:math:`(x_{min}, {\\infty})`.Moreover, some real-world systems may have a
power-law regime in which case the distribution is bounded from both above and
below, and will have the support :math:`(x_{min}, x_{max})`.

Assuming a continuous random variable distributed according to a power law
with exponent :math:`\\alpha > 1`, the PDFs for the two cases above are
estimated by following the steps in [1]_, using which the cumulative
distribution function (CDF) for the two cases are also easily estimated. The
functional forms of the PDFs and the CDFs are given below, and are also
implemented in this module as :py:func:`pdf` and :py:func:`cdf`.

In order to generate random samples for a power-law distributed random
variable, this module uses the inverse transform method [2]_ which relies on the
inverse function of the CDF of the random variable under consideration. The
inverse funcational form of the CDfs of the two power-law distribution types
considered here are also listed below. The random sampling routine is
implemented in :py:func:`sample`.

Formulae
--------
1. Power-law distributed on :math:`(x_{min}, \\infty)`:
+++++++++++++++++++++++++++++++++++++++++++++++++++++++

    PDF:

    .. math::
        p(x) = - \\frac{\\beta}{x_{min}^{\\beta}} x^{-\\alpha}

    CDF:

    .. math::
        P(x) = - \\frac{x_{min}^{\\beta} - x^{\\beta}}{x_{min}^{\\beta}}

    Inverse of CDF:

    .. math::
        P^{-1}(x) = - x_{min} (1 - x)^{1 / \\beta}

2. Power-law distributed on :math:`(x_{min}, x_{max})`
++++++++++++++++++++++++++++++++++++++++++++++++++++++

    PDF:

    .. math::
        p(x) = \\frac{\\beta}{x_{max}^{\\beta} - x_{min}^{\\beta}}x^{-\\alpha}

    CDF:

    .. math::
        P(x) = \\frac{x^{\\beta} - x_{min}^{\\beta}}
                     {x_{max}^{\\beta} - x_{min}^{\\beta}}.

    Inverse of CDF:

    .. math::
        P^{-1}(x) &= (a + bx) ^ {1 / \\beta}

        a &= x_{min}^{\\beta}

        b &= x_{max}^{\\beta} - a

where we have defined :math:`\\beta = 1 - \\alpha` for a more concise notation.
Note that the inverse transform method used here recommended by Clauset et al.
in Sec. II B of [1]_ in order to generate random samples according the given
power-law exponent. Also, note that they provide a separate set of formulae
for discrete power-law variables which are not considered here. Thus, care
should be taken before extending the results of code implemented in this module
to discrete power-law distributed random variables.


References
----------
.. [1]  Clauset, A., Shalizi, C. R., Newman, M. E. J.
        "Power-law Distributions in Empirical Data".
        SFI Working Paper: 2007-12-049 (2007).
        http://www.santafe.edu/media/workingpapers/07-12-049.pdf

.. [2]  "Inverse transform sampling".
        Wikipedia. Accessed on: 23 May, 2016.
        https://en.wikipedia.org/wiki/Inverse_transform_sampling

"""
# Created: Mon May 23, 2016  02:56PM
# Last modified: Wed May 25, 2016  11:41AM
# Copyright: Bedartha Goswami <goswami@uni-potsdam.de>


import numpy as np
import matplotlib.pyplot as pl

__all__ = ["sample", "pdf", "cdf", "histogram"]


def sample(alpha=2.5, size=1000, xmin=1, xmax=None):
    """
    Generates a random sample from a (bounded) power-law distribution.

    Parameters
    ----------
    alpha : scalar, float, greater than 1
        power-law exponent
    size : scalar, integer
        size (i.e. length) of the random sample to be generated
    xmin : scalar, float
        lower bound of the power-law random variable (greater than zero)
    xmax : scalar, float, optional
        upper bound of the power-law random variable

    Returns
    -------
    s : numpy.ndarray with ``shape = (size, )``
        random deviates from a power-law distribution with exponent ``alpha``
        and with upper and lower bounds as specified

    See Also
    --------
    histogram
    pdf

    """
    assert alpha > 1, "Power law exponent should be greater than 1!"
    beta = 1. - alpha
    r = np.random.rand(size)
    if not xmax:
        s = xmin * ((1. - r) ** (1 / beta))
    else:
        a = xmin ** beta
        b = xmax ** beta - a
        s = (a + b * r) ** (1. / beta)
    return s


def pdf(x, alpha, xmin, xmax=None):
    """
    Returns the PDF of a (bounded) power-law variable.

    Parameters
    ----------
    x : numpy.ndarray of 1-dimension
        array of values on which the power-law PDF is to be evaluated
    alpha : scalar, float, greater than 1
        power-law exponent
    xmin : scalar, float
        lower bound of the power-law random variable (greater than zero)
    xmax : scalar, float, optional
        upper bound of the power-law random variable

    Returns
    -------
    pdf : numpy.ndarray with ``shape = (len(x), )``
        array of the probability densities evaluated for each point in ``x``.

    See Also
    --------
    cdf
    sample

    """
    assert alpha > 1, "Power law exponent should be greater than 1!"
    beta = 1. - alpha
    if not xmax:
        k = - beta / xmin ** beta
        pdf = k * (x ** -alpha)
    else:
        k = xmax ** beta - xmin ** beta
        pdf = (beta / k) * (x ** -alpha)
    return pdf


def cdf(x, alpha, xmin, xmax=None):
    """
    Returns the CDF of a (bounded) power-law variable.

    Parameters
    ----------
    x : numpy.ndarray of 1-dimension
        array of values on which the power-law CDF is to be evaluated
    alpha : scalar, float, greater than 1
        power-law exponent
    xmin : scalar, float
        lower bound of the power-law random variable (greater than zero)
    xmax : scalar, float, optional
        upper bound of the power-law random variable

    Returns
    -------
    cdf : numpy.ndarray with ``shape = (len(x), )``
        array of the cumulative probabilities evaluated for each point in ``x``.

    See Also
    --------
    pdf
    sample

    """
    assert alpha > 1, "Power law exponent should be greater than 1!"
    beta = 1. - alpha
    if not xmax:
        k = 1.
        cdf = k - (x / xmin) ** beta
    else:
        k = xmax ** beta - xmin ** beta
        cdf = (x ** beta - xmin ** beta) / k
    return cdf


def histogram(arr, bins=None, plot=False, ax=None, **kwargs):
    """
    Plots a power law histogram using logarithmic binning.

    Parameters
    ----------
    arr : numpy.ndarray of 1-dimension
        sample of a power-law distributed random deviates
    bins : numpy.ndarray of 1-dimension, optional
        bins for which the histogram counts are obtained. This should be of
        ``shape = (len(arr) + 1, )``. If not provided, an array of
        logarithmically spaced bins is estimated using Sturges' formula [3]_.
    plot : boolean
        If ``True``, the histogram results are plotted on given axes or on a
        newly created ``matplotlib.axes`` object. Default ``= False``.
    ax : matplotlib.axes, optional
        axes on which the estimated histogram is to be plotted. If not
        provided, a standard ``matplotlib.axes`` object is created for plotting.

    Returns
    -------
    hist : numpy.ndarray of 1 dimension
        array containing the histogram counts/fractions for the given random
        sample. This is of length ``n``, where ``n`` is the number of bins
        given by Sturges formula.
    bins : numpy.ndarray of 1 dimension
        array containing the bin edges for the given random sample obtained by
        constructing ``n`` number of logarithmically spaced bins.
    ax : matplotlib.axes
        axes on which the histogram is plotted

    Notes
    -----
    The **kwargs** arguments are partly passed on to ``numpy.histogram`` (such
    as the ``density`` keyword argument), and to ``matplotlib.pyplot.plot``
    (such as the ``mec``, ``mfc``, and ``ms`` keywords), and finally to also
    specify axes labels and their fontsizes using ``xlab`` and ``fs`` keyword
    arguments. These could also, in principle, be done separately outside of
    this function by using the necessary methods of the ``ax`` object.

    See Also
    --------
    pdf
    sample

    References
    ----------
    .. [3]  Sturges, H. A. (1926).
            "The choice of a class interval".
            Journal of the American Statistical Association, 21(153), 65-66.
            http://www.tandfonline.com/doi/abs/10.1080/01621459.1926.10502161

    """
    # argument parsing
    density = False
    mec, mfc, ms = "none", "Black", 5
    xlab, fs = "Observable (Units)", 12
    for key in kwargs:
        exec("%s = kwargs[key]" % key)
    if not bins:
        bins = _logbins(arr)
    # histogram counts
    hist = np.histogram(arr, bins=bins, density=density)[0]

    # histogram plot
    if plot:
        # create axes if not given
        if not ax:
            fig = pl.figure(figsize=[8.5, 6.5])
            ax = fig.add_subplot(111)
        # plot histogram results
        mid_pts = 0.5 * (bins[:-1] + bins[1:])
        ax.plot(mid_pts, hist, "o",
                mec=mec, mfc=mfc, ms=ms)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(which="both", axis="both")
        ax.set_xlabel(xlab, fontsize=fs)
        ax.set_ylabel("Probability", fontsize=fs)

    # return histogram results and axis object for further plotting
    if plot:
        return hist, bins, ax
    else:
        return hist, bins


def _logbins(arr):
    """
    Returns an array of logarithmically spaced bins for given array.
    """
    n = len(arr)
    nbins = np.ceil(np.log2(n)) + 1         # Sturges formula
    arrmin = arr[arr != 0.].min()
    arrmax = arr.max()
    bins = np.logspace(np.log10(arrmin), np.log10(arrmax),
                       num=nbins, base=10)
    return bins
