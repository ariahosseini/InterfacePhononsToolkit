"""
A method to compute the auto-correlation function(acf) using different methods
"""

import numpy as np


def acf(time: np.ndarray, flux: np.ndarray) -> np.ndarray:

    """
    Compute the auto-correlation of the instantaneous heat flux vector

    :arg
        time: np.ndarray
            Correlation time lag
        flux: np.ndarray
            Instantaneous heat flux vector
    :returns
        auto_correlation: np.ndarray
            : Auto-correlation
    """

    time_intervals = time.size
    auto_correlation = np.array([[np.dot(flux[j, :time_intervals - i], flux[j, i:])
                     for i in range(time_intervals)] for j in range(3)])
    auto_correlation *= (time[-1] - time[0]) / float(time_intervals)

    return auto_correlation


def acf_fft(time: np.ndarray, flux: np.ndarray) -> np.ndarray:

    """
    Compute auto-correlation of the instantaneous flux vector using fast fourier transform.
    To get rid of the assumed periodicity in the ACF using this method, and the symmetry that one gets in the result,
    the length of the vector is doubled and fill the 2nd half with zeros. The first half of the result is taken.

    :arg
        time: np.ndarray
            Correlation time lag
        flux: np.ndarray
            Instantaneous heat flux vector
    :returns
        auto_correlation: np.ndarray
            : Auto-correlation
    """

    time_intervals = np.shape(flux)[1]
    acf = np.zeros([3, time_intervals * 2])  # Auto-correlation

    dt = time[-1] / float(time_intervals)  # Time-step
    for j in range(3):
        dft = np.fft.fft(np.concatenate([flux[j], np.zeros(time_intervals)]))
        acf[j] = np.fft.ifft(dft * np.conjugate(dft)) * dt
    return acf[:, :time_intervals]


def acf_numpy(flux: np.ndarray) -> np.ndarray:

    """
    # Compute the auto-correlation of the instantaneous flux vector using numpy correlation command
    :arg
        flux: np.ndarray
            Instantaneous heat flux vector
    :returns
        auto_correlation: np.ndarray
            : Auto-correlation
    """

    auto_correlation = np.array([np.correlate(flux[i]) for i in range(3)])

    return auto_correlation
