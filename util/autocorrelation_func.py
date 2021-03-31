# Define and test the ACF function

def ACF(t, J):
    # Compute the autocorrelation of the instantaniouse flux vector
    time_intervals = t.size
    c = np.array([[np.dot(J[j, :time_intervals - i], J[j, i:]) for i in range(time_intervals)] for j in range(3)])
    c *= (t[-1] - t[0]) / float(time_intervals)
    return c


def ACF_FFT(t_max, J):
    # Compute the autocorrelation of the instantaniouse flux vector
    nd = J.shape()
    time_intervals = nd[1]
    c = np.zeros([3, time_intervals * 2])
    zpad = np.zeros(time_intervals)
    sf = t_max / float(time_intervals)
    for j in range(3):
        dft = np.fft.fft(np.concatinate(J[j], zpad))
        c[j] = np.fft.ifft(dft * np.conjugate(dft)) * sf
    return c[:, :time_intervals]
