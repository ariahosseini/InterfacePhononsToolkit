"""
autocorrelation_func.py compute the autocorrelation (ACF) using different methods

Author: S. Aria Hosseini
Email: shoss008@ucr.edu
"""

import numpy as np


def ACF(time, J):

    """

    Compute the autocorrelation of the instantaneous heat flux vector

    :arg
        time                                 : Autocorrelation time lag
        J                                    : Instantaneous heat flux vector, Numpy array of 3 by np.shape(time)
    :returns
        acf                                  : Autocorrelation
    """

    time_intervals = time.size
    acf = np.array([[np.dot(J[j, :time_intervals - i], J[j, i:]) for i in range(time_intervals)] for j in range(3)])
    acf *= (time[-1] - time[0]) / float(time_intervals)   # Autocorrelation
    return acf


def ACF_FFT(time, J):

    """

    Compute autocorrelation of the instantaneous flux vector using fast fourier transform.

    To get rid of the assumed periodicity in the ACF using this method, and the symmetry that one gets in the result,
    the length of the vector is doubled and fill the 2nd half with zeros. The first half of the result is taken.

    :arg
        time                                 : Autocorrelation time lag
        J                                    : Instantaneous heat flux vector, Numpy array of 3 by np.shape(time)
    :returns
        acf                                  : Autocorrelation

    """

    time_intervals = np.shape(J)[1]
    acf = np.zeros([3, time_intervals * 2])   # Autocorrelation   # Autocorrelation

    dt = time[-1] / float(time_intervals)
    for j in range(3):
        dft = np.fft.fft(np.concatenate([J[j], np.zeros(time_intervals)]))
        acf[j] = np.fft.ifft(dft * np.conjugate(dft)) * dt
    return acf[:, :time_intervals]


def ACF_NP(J):

    # Compute the autocorrelation of the instantaneous flux vector using numpy correlation command

    acf = [np.correlate(J[i]) for i in range(3)]

    return acf   # Autocorrelation


def _ACF(time, J):

    """

    Compute the autocorrelation of the instantaneous heat flux vector.
    Same opperation as ACF, but laid out differently

    :arg
        time                                 : Autocorrelation time lag
        J                                    : Instantaneous heat flux vector, Numpy array of 3 by np.shape(time)
    :returns
        acf                                  : Autocorrelation
    """

    acf = 0 * J
    time_intervals = time.size
    for j in range(3):
        for i in range(time_intervals):
            acf[j, i] = np.dot(J[j, :time_intervals - i], J[j, i:])
        acf *= (time[-1]-time[0])/float(time_intervals)

    return acf
