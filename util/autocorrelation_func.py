"""
autocorrelation_func.py compute the autocorrelation (ACF) using defferent methods

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


def ACF_FFT(t_max, J):

    # Compute the fast fourier transform of the instantaneous flux vector autocorrelation

    nd = np.shape(J)
    print(nd)
    time_intervals = nd[1]
    c = np.zeros([3, time_intervals * 2])
    zpad = np.zeros(time_intervals)
    print(np.shape(np.concatenate([J[0], zpad])))
    sf = t_max / float(time_intervals)
    for j in range(3):
        dft = np.fft.fft(np.concatenate([J[j], zpad]))
        c[j] = np.fft.ifft(dft * np.conjugate(dft)) * sf
    return c[:, :time_intervals]


def ACF_NP(t, J):

    # Compute the autocorrelation of the instantaneous flux vector using numpy correlation command

    acf = [np.correlate(J[i]) for i in range(3)]
    return acf   # Autocorrelation


def ACF2(time, J):

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
