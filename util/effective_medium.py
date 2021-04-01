"""
effective_medium.py is a collection of methods to compute effective thermal conductivity and figure of merits in
dielectrics containing nano- and micro-scale pores. the pores are assumed to be aligned and cylindrical in shape.

Author: S. Aria Hosseini
Email: shoss008@ucr.edu
"""


import numpy as np
from scipy.optimize import curve_fit


def logistic(mfp, lambda_o):

    """
    cumulative lattice thermal conductivity of bulk dielectrics is often modeled as logistic regression of form:
    cumulative_kappa = kappa_bulk/ (1+lambda_o/mean_free_path).

    We use curve_fit from scipy.optimize to fit the logistic function to first-principle cumulative kappa generated
    using almaBTE package. Here, lambda-o is the fitting parameter.

    :arg
        mfp                                 : An array of sorted phonon mean free path in unit of m
        lambda_o                            : The fitting parameter in unit of m

    :returns
        A logistic curve fitted to the cumulative_kappa vs. phonon mean free path in unit of W/m/K

    """

    return kappa_bulk / (1 + lambda_o / mfp)

"""

Following methods are to compute the phonon lifetime in dielectrics containing nano-scale pores with cylindrical shape.

See following articles for the details

1) "Modified effective medium formulation for the thermal conductivity of nanocomposites" for Minnich func.
2) "Size and porosity effects on thermal conductivity of nanoporous material with an extension to nanoporous particles
    embedded in a host matrix" for Machrafi func.
3) "Thermal conductivity modeling of micro- and nanoporous silicon" for Liu func.

    :arg
        phi                           : Porosity, unitless
        Lp                            : Pore-pore spacing, im m

    :returns
        Lc                            : Phonon mean free path in porous structures, in m

"""


def Minnich(phi, Lp):   # Minnich et al., Applied Physics Letter, 2007

    """
    :param phi: Porosity, unitless
    :param Lp: Pore-pore spacing, im m
    :return Lc: Phonon mean free path in porous structures, in m

    """

    Lc = (1 - phi.T) / np.sqrt(4 * phi.T / np.pi) * Lp

    return Lc


def Machrafi(phi, Lp):   # Machrafi et al., Physics Letters A, 2015

    """
    :param phi: Porosity, unitless
    :param Lp: Pore-pore spacing, im m
    :return Lc: Phonon mean free path in porous structures, in m

    """

    Lc = 1 / np.sqrt(4 * phi.T / np.pi) * Lp

    return Lc


def Liu(phi, Lp):   # Liu et al., International Journal of Thermal Sciences, 2010

    """
    :param phi: Porosity, unitless
    :param Lp: Pore-pore spacing, im m
    :return Lc: Phonon mean free path in porous structures, in m

    """

    Lc = (1 - phi.T) * np.sqrt(1 / np.sqrt(phi.T / np.pi) - 1) * Lp + phi.T * 2 * np.sqrt(phi.T / np.pi) * Lp

    return Lc


def Maxwell_Garnett(phi):

    """
    :param phi: Porosity, unitless
    :return So: Maxwell Garnett porosity function

    """

    So = (1 - phi) / (1 + phi)

    return So


def Maxwell_Eucken(phi):

    """
    :param phi: Porosity, unitless
    :return So: Maxwell Eucken porosity function

    """

    So = (1 - phi) / (1 + phi / 2)

    return So


def Rayleigh(phi):

    """
    :param phi: Porosity, unitless
    :return  So: Rayleigh porosity function

    """

    So = 1. / 3 * (2 * (1 - 2 * phi / (1 + phi + 0.30584 * np.power(phi, 4) + 0.013363 * np.power(phi, 8))) + (1 - phi))

    return So


def view_factor(phi):

    """
    :param phi: Porosity, unitless
    
    :return
        F_c:  phonon view factor in materials with cylindrical pores
        F_s:  phonon view factor in materials with cubic pores
        F_t:  phonon view factor in materials with triangle prism pores

    """

    F_c = 1 - np.sqrt(4*phi/np.pi)*(np.pi/2-(np.arcsin(np.sqrt(4*phi/np.pi))+np.sqrt(np.pi/4/phi-1)-
                                             np.sqrt(np.pi/4/phi)))

    F_s = 1-2*np.sqrt(phi)*(1-1/2*(np.sqrt(1+(1/np.sqrt(phi)-1)**2)-(1/np.sqrt(phi)-1)**2))

    F_t = np.sqrt(4*phi/np.sqrt(3)-2*np.sqrt(phi/np.sqrt(3))+1)-2*np.sqrt(phi/np.sqrt(3))

    return F_c, F_s, F_t


def bulk_mean_free_path(path_kappa_cumulative, maxfev=1000):

    """
    This function compute the logistic regressionn fit to the cumulative thermal conductivity computed in
    first-principle thermal conductivity simulator, almaBTE

    cumulative lattice thermal conductivity of bulk dielectrics is often modeled as logistic regression of form:
    cumulative_kappa = kappa_bulk/ (1+lambda_o/mean_free_path).


    :arg
        path_kappa_cumulative             : point to the cumulative kappa file
        maxfev                            : maxfev, maximum iteration for fitting

    :returns
        lambda_bulk                       : The uniparameter from logistic regression

    """

    cumulative_data = np.loadtxt(path_kappa_cumulative, skiprows=1, delimiter=',')  # Cumulative thermal conductivity
    lambda_bulk, _ = curve_fit(Logistic, cumulative_data[:, 0], cumulative_data[:, 1], maxfev=maxfev)   # Logistic fit

    return lambda_bulk


def kappa_effective(So, Lc, lambda_bulk):

    """
    This function compute the effective thermal conductivity of dielectrics containing nanoscale porosity


    :arg
        So                              : Porosity function, unitless
        Lc                              : Phonon mean free path in porous structures, in m
        lambda_bulk                    : The uniparameter from logistic regression

    :returns
         Kn                              : Knudsen number, unitless
         zeta                            : A descriptor equal to (1+Kn*(Ln(Kn)-1))/(kn-1)^2
         kappa_effective                 : Effective thermal conductivity of dielectrics containing nanoscale porosity

    """

    Kn = lambda_bulk / Lc
    zeta = (1 + Kn * (np.log(Kn) - 1)) / (Kn - 1)**2
    kappa_effective = kappa_bulk * So.T * zeta

    return Kn, zeta, kappa_effective


def fractional_effective_ZT(Kn, phi):

    """
    This function compute the ZT_porous/ ZT_bulk in dielectrics containing nanoscale porosity


    :arg
        Kn                              : Knudsen number, unitless
        phi                             : Pprosity

    :returns
         fractional_ZT_effective        : ZT_porous/ ZT_bulk as a function of Knudsen number for the given porosity

    """

    zeta = (1 + Kn * (np.log(Kn) - 1)) / (Kn - 1)**2
    fractional_ZT_effective = (1+phi.T)/zeta

    return fractional_ZT_effective
