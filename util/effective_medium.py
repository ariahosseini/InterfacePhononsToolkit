import numpy as np
import math
from scipy.optimize import curve_fit
import os


def Logistic(mfp, lc):
    return S_diffusive / (1 + mfp / lc)


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

    lambda_bulk = np.array([])

    for porosity_index in range(np.shape(phi)[1]):
        global S_diffusive
        S_diffusive = So[0, porosity_index]
        _lambda_bulk, _ = curve_fit(Logistic, cumulative_data[:, 0], cumulative_data[:, 1], maxfev=maxfev)
        lambda_bulk = np.append(lambda_bulk, _lambda_bulk)
        del S_diffusive

    return lambda_bulk


path_kappa_cumulative = '../kappa_almabte/GaN_30_30_30_1,0,0_800K.kappacumul_MFP'
# cumulative_data = np.loadtxt(path_kappa_cumulative, skiprows=1, delimiter=',')
# maxfev = 1000

# lambda_bulk, a = curve_fit(Logistic, cumulative_data[:, 0], cumulative_data[:, 1], maxfev=maxfev)

# print(lambda_bulk)

phi = np.array([[0.05, 0.25, 0.40, 0.55, 0.70, 0.15]])
Lp = np.array([[40, 80, 100, 500, 1000, 4000, 10000]])

So = Rayleigh(phi=phi)
Lc = Liu(phi=phi, Lp=Lp)
# print(np.shape(Lc))

# lambda_bulk = np.array([])

# for porosity_index in range(np.shape(phi)[1]):

#     global S_diffusive
#     S_diffusive = So[0, porosity_index]
#     _lambda_bulk, _ = curve_fit(Logistic, cumulative_data[:, 0], cumulative_data[:, 1], maxfev=maxfev)
#     lambda_bulk = np.append(lambda_bulk, _lambda_bulk)
#     del S_diffusive


Lambda_o = bulk_mean_free_path(path_kappa_cumulative=path_kappa_cumulative, phi=phi, So=So, maxfev=1000)
print(Lambda_o)


exit()
