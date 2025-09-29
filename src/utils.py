"""Module for utility functions for conducting selective inference."""

import numpy as np
from scipy.special import logsumexp  # type: ignore[import]
from scipy.stats import rv_continuous  # type: ignore[import]
from sicore import RealSubset  # type: ignore[import]


def compute_log_area(rv: rv_continuous, intervals: RealSubset) -> float:
    """Compute the logarithm of the integral of the pdf over the each interval.

    Parameters
    ----------
    rv : rv_continuous
        The rv_continuous instance to be integrated.
    intervals : RealSubset
        The intervals on which to compute the integral.

    Returns
    -------
    float
        The logarithm of the integral.
    """
    left_ends, right_ends = intervals.intervals.T
    log_each_area = np.empty(len(intervals))
    mask = left_ends < rv.median()

    left_log_cdf, right_log_cdf = (
        rv.logcdf(left_ends[mask]),
        rv.logcdf(right_ends[mask]),
    )
    log_each_area[mask] = right_log_cdf + _log1mexp(left_log_cdf - right_log_cdf)

    left_log_sf, right_log_sf = rv.logsf(left_ends[~mask]), rv.logsf(right_ends[~mask])
    log_each_area[~mask] = left_log_sf + _log1mexp(right_log_sf - left_log_sf)

    return logsumexp(log_each_area)


def _log1mexp(z: np.ndarray) -> np.ndarray:
    """Compute the logarithm of one minus the exponential of the input array, element-wise.

    Parameters
    ----------
    z : np.ndarray
        Input values.

    Returns
    -------
    np.ndarray
        Logarithm of one minus the exponential of the input array.
    """
    z = np.asarray(z)
    values = np.empty_like(z)
    halflog = -0.693147  # equal to log(0.5)
    mask = z < halflog
    values[mask] = np.log1p(-np.exp(z[mask]))
    values[~mask] = np.log(-np.expm1(z[~mask]))
    return values
