import numpy as np
from scipy.optimize import curve_fit


def Logistic(mfp, lambda_o):
    return kappa_bulk / (1 + lambda_o / mfp)


def Minnich(phi, Lp):
    Lc = (1 - phi.T) / np.sqrt(4 * phi.T / np.pi) * Lp
    return Lc


def Machrafi(phi, Lp):
    Lc = 1 / np.sqrt(4 * phi.T / np.pi) * Lp
    return Lc


def Liu(phi, Lp):
    Lc = (1 - phi.T) * np.sqrt(1 / np.sqrt(phi.T / np.pi) - 1) * Lp + phi.T * 2 * np.sqrt(phi.T / np.pi) * Lp
    return Lc


def Maxwell_Garnett(phi):
    So = (1 - phi) / (1 + phi)
    return So


def Maxwell_Eucken(phi):
    So = (1 - phi) / (1 + phi / 2)
    return So


def Rayleigh(phi):
    So = 1. / 3 * (2 * (1 - 2 * phi / (1 + phi + 0.30584 * np.power(phi, 4) + 0.013363 * np.power(phi, 8))) + (1 - phi))
    return So

def bulk_mean_free_path(path_kappa_cumulative, phi, So, maxfev=1000):

    cumulative_data = np.loadtxt(path_kappa_cumulative, skiprows=1, delimiter=',')
    lambda_bulk, _ = curve_fit(Logistic, cumulative_data[:, 0], cumulative_data[:, 1], maxfev=maxfev)

    return lambda_bulk

def kappa_effective(So, Lc, Lambda_o):

    Kn = Lambda_o / Lc
    zeta = (1 + Kn * (np.log(Kn) - 1)) / (Kn - 1)**2
    kappa_effective = kappa_bulk * So.T * zeta

    return Kn, zeta, kappa_effective
