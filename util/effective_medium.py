import numpy as np
import math
from scipy.optimize import curve_fit
import os


def Logistic(mfp, lc):
    return So / (1 + mfp / lc)


def Minnich(phi, Lp):
    Lc = (1 - phi.T) / np.sqrt(4 * phi.T / np.pi) * Lp
    return Lc


def Machrafi(phi, Lp):
    Lc = 1 / np.sqrt(4 * phi.T / np.pi) * Lp
    return Lc


def Liu(phi, Lp):
    Lc = (1 - phi.T) * np.sqrt(1 / np.sqrt(phi.T / np.pi) - 1) * Lp + phi.T * 2 * np.sqrt(phi.T / np.pi) * Lp
    return Lc


def Maxwell_Garnett(phi, Lp):
    So = (1 - phi) / (1 + phi)
    lambda_inside = 2 * r
    lambda_outside = sqrt()
    return So


def Maxwell_Eucken(phi, Lp):
    So = (1 - phi) / (1 + phi / 2)
    return So


def Rayleigh(phi, Lp):
    So = 1. / 3 * (2 * (1 - 2 * phi / (1 + phi + 0.30584 * np.power(phi, 4) + 0.013363 * np.power(phi, 8))) + (1 - phi))
    return So


def Triangle(phi, Lp):
    So = -4.37 * phi**3 + 3.47 * phi**2 - 2.67 * phi + 1
    return So


def Tall_rectangular(phi, Lp):
    So = -2.27 * phi**3 + 2.97 * phi**2 - 2.61 * phi + 1
    return So


def Flat_rectangular(phi, Lp):
    So = -2.09 * phi**3 + 2.924 * phi**2 - 1.8 * phi + 1
    return So


path_kappa_cumulative = 'kappa_almabte/GaN_30_30_30_1,0,0_800K.kappacumul_MFP'
cumulative_data = np.loadtxt('path_kappa_cumulative', skiprows=1, delimiter=',')
maxfev = 1000

popt, pcov = curve_fit(Logistic, cumulative_data[:, 0], cumulative_data[:, 1], maxfev=maxfev)

phi = np.array([[0.05, 0.25, 0.40, 0.55, 0.70, 0.15]])
Lp = np.array([[40, 80, 100, 500, 1000, 4000, 10000]])


# i: porosity index 0:005, 1:0.25, 2:040, 3:0.55, 4:070, 5:0.15
# j: pore pore spacing index 0:40nm, 1:80nm, 2:100nm, 3:500nm, 4:1000nm, 5:4000, 6:10000nm
for i in range(6):
    for j in range(7):

        if i == 0:
            S = S_005
        elif i == 1:
            S = S_025
        elif i == 2:
            S = S_040
        elif i == 3:
            S = S_055
        elif i == 4:
            S = S_070
        elif i == 5:
            S = S_015

        global So
        # So = (1 - phi[0, i]) / (1 + phi[0, i])
        So = S[0, j]
        popt, pcov = curve_fit(Logistic, mfp.flatten(), S[:, j], maxfev=1000)
