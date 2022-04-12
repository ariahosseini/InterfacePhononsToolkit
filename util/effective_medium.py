"""
A collection of methods to compute effective thermal conductivity of nano-structures.
"""

import numpy as np
from scipy.optimize import curve_fit


def bulk_mean_free_path(path_kappa_cumulative: 'str', max_fev: int = 1000) -> float:

    """
    This "bulk_mean_free_path" function computes the logistic fit to the cumulative thermal conductivity
    computed in first-principle thermal conductivity simulator, almaBTE with the following form:
    cumulative_kappa = kappa_bulk/ (1+lambda_o/mean_free_path).
    :arg
        path_kappa_cumulative: 'str'
            Point to the cumulative kappa file
        max_fev: int
            Maximum iteration for fitting
    :returns
        lambda_bulk: float
            The uni-parameter characteristic mfp
    """

    def logistic_curve(mfp, lambda_o):

        """
        :arg
            mfp: np.ndarray
                Phonon mean free path [m]
            lambda_o: float
                The fitting parameter [m]
        :returns
            fit_curve: np.ndarray
                Logistic fit to the cumulative_kappa vs. phonon mean free path [W/m/K]
        """

        fit_curve = kappa_bulk / (1 + lambda_o / mfp)

        return fit_curve

    cumulative_data = np.loadtxt(path_kappa_cumulative, skiprows=1, delimiter=',')  # Cumulative thermal conductivity
    kappa_bulk = cumulative_data[-1, 1]  # Bulk thermal conductivity
    lambda_bulk, _ = curve_fit(logistic_curve, cumulative_data[:, 0],
                               cumulative_data[:, 1], maxfev=max_fev)  # Characteristic mean-free-path

    return lambda_bulk


def minnich_model(phi: np.ndarray, len_p: np.ndarray) -> np.ndarray:

    """
    :param
        phi: np.ndarray
        Porosity [unit-less]
        len_p: np.ndarray
            pore-pore spacing [m]
    :return
        len_c: np.ndarray
            Characteristic length in porous structures [m]
    """

    len_c = (1 - phi.T) / np.sqrt(4 * phi.T / np.pi) * len_p

    return len_c


def machrafi_model(phi: np.ndarray, len_p: np.ndarray) -> np.ndarray:

    """
    :param
        phi: np.ndarray
        Porosity [unit-less]
        len_p: np.ndarray
            pore-pore spacing [m]
    :return
        len_c: np.ndarray
            Characteristic length in porous structures [m]
    """

    len_c = 1 / np.sqrt(4 * phi.T / np.pi) * len_p

    return len_c


def liu_model(phi: np.ndarray, len_p: np.ndarray) -> np.ndarray:

    """
    :param
        phi: np.ndarray
        Porosity [unit-less]
        len_p: np.ndarray
            pore-pore spacing [m]
    :return
        len_c: np.ndarray
            Characteristic length in porous structures [m]
    """

    len_c = (1 - phi.T) * np.sqrt(1 / np.sqrt(phi.T / np.pi) - 1) * len_p \
            + phi.T * 2 * np.sqrt(phi.T / np.pi) * len_p

    return len_c


def maxwell_garnett_model(phi: np.ndarray) -> np.ndarray:

    """
    :param
        phi: np.ndarray
            Porosity [unit-less]
    :return
        So: np.ndarray
            Maxwell Garnett model of macroscopic suppression function
    """

    So = (1 - phi) / (1 + phi)

    return So


def maxwell_eucken(phi):

    """
    :param
        phi: np.ndarray
            Porosity [unit-less]
    :return
        So: np.ndarray
            Maxwell Eucken model of macroscopic suppression function
    """

    So = (1 - phi) / (1 + phi / 2)

    return So


def Rayleigh_model(phi: np.ndarray) -> np.ndarray:

    """
    :param
        phi: np.ndarray
            Porosity [unit-less]
    :return
        So: np.ndarray
            Rayleigh model of macroscopic suppression function
    """

    So = 1. / 3 * (2 * (1 - 2 * phi / (1 + phi + 0.30584 * phi**4 + 0.013363 * phi**8)) + (1 - phi))

    return So


def view_factor(phi: np.ndarray) -> dict:

    """
    :param
        phi: np.ndarray
            Porosity [unit-less]
    :return
    view_factors: dict
            view_factors['cylindrical']:  Phonon view factor in materials with cylindrical pores
            view_factors['cubic']:  Phonon view factor in materials with cubic pores
            view_factors['triangle_prism']:  Phonon view factor in materials with triangle prism pores
    """

    F_cy = 1 - np.sqrt(4 * phi / np.pi) * (
                np.pi / 2 - (np.arcsin(np.sqrt(4 * phi / np.pi)) + np.sqrt(np.pi / 4 / phi - 1) -
                             np.sqrt(np.pi / 4 / phi)))

    F_cu = 1 - 2 * np.sqrt(phi) * (1 - 1 / 2 * (np.sqrt(1 + (1 / np.sqrt(phi) - 1) ** 2) - (1 / np.sqrt(phi) - 1) ** 2))

    F_tri = np.sqrt(4 * phi / np.sqrt(3) - 2 * np.sqrt(phi / np.sqrt(3)) + 1) - 2 * np.sqrt(phi / np.sqrt(3))

    view_factors = {'cylindrical': F_cy, 'cubic': F_cu, 'triangle_prism': F_tri}

    return view_factors


def kappa_eff_single_dof(kappa_bulk: np.ndarray, So: np.ndarray, len_c: np.ndarray, lambda_bulk: np.ndarray) -> dict:

    """
    This function compute the effective thermal conductivity of nano-engineered dielectrics
    :arg
        kappa_bulk: np.ndarray
            Bulk thermal conductivity [W/mK]
        So: np.ndarray
            Macroscopic suppression function
        len_c: np.ndarray
            Characteristic length in porous structures [m]
        lambda_bulk: np.ndarray
            The Characteristic mfp [m]
    :returns
        output: dict
         output['Kn']: np.ndarray
            Knudsen number [unit-less]
         output['correction_term']: np.ndarray
            Ballistic correction term equal to (1+Kn*(Ln(Kn)-1))/(Kn-1)^2
         output['kappa_effective']: np.ndarray
            Thermal conductivity
    """

    Kn = lambda_bulk / len_c
    zeta = (1 + Kn * (np.log(Kn) - 1)) / (Kn - 1) ** 2
    kappa_effective = kappa_bulk * So.T * zeta

    output = {'Kn': Kn, 'Ballistic Correction Term': zeta, 'Thermal conductivity': kappa_effective}

    return output


def kappa_eff_multi_dof(kappa_bulk: np.ndarray, So: np.ndarray, len_c: np.ndarray, lambda_bulk: np.ndarray) -> dict:

    """
    This function compute the effective thermal conductivity of nano-engineered
    dielectrics with multiple degrees of confinements
    :arg
        kappa_bulk: np.ndarray
            Bulk thermal conductivity [W/mK]
        So: np.ndarray
            Macroscopic suppression function
        len_c: np.ndarray
            Characteristic length in porous structures [m]
        lambda_bulk: np.ndarray
            The Characteristic mfp [m]
    :returns
        output: dict
         output['Kn']: np.ndarray
            Knudsen numbers [unit-less]
         output['correction_term']: np.ndarray
            Ballistic correction term equal to Kn1^2*Ln(Kn1)/(Kn1-1)^2/(Kn1-Kn2) +
                                               Kn2^2*Ln(Kn2)/(Kn2-1)^2/(Kn2-Kn1) +
                                               1/(Kn1-1)/(Kn2-1)
         output['kappa_effective']: np.ndarray
            Thermal conductivity
    """

    Kn_1 = lambda_bulk / len_c[0]
    Kn_2 = lambda_bulk / len_c[1]

    zeta = Kn_1**2*np.log(Kn_1)/(Kn_1-1)**2/(Kn_1-Kn_2) + \
           Kn_2**2*np.log(Kn_2)/(Kn_2-1)**2/(Kn_2-Kn_1) + \
           1/(Kn_1-1)/(Kn_2-1)

    kappa_effective = kappa_bulk * So.T * zeta

    output = {'Kn': np.array([Kn_1, Kn_2]), 'Ballistic Correction Term': zeta, 'Thermal conductivity': kappa_effective}

    return output


def kappa_bulk(path_phononinfo: str) -> float:

    """
    This function compute the bulk lattice thermal conductivity from AlmaBTE phononinfo output
    :arg
        path_phononinfo: str
    :returns
        kappa: float
            Thermal conductivity
    """

    data = np.loadtxt(path_phononinfo, skiprows=1, delimiter=',')
    tau_p = data[:, 7]  # Phonon lifetime [s]
    vel = data[:, 8:]  # Phonon group velocity [m/s]
    Cv = data[:, 6]  # Phonon specific heat [J/m^3-K]

    kappa = np.einsum('ki,kj,k,k->ij', vel, vel, tau_p, Cv)  # Lattice thermal conductivity [W/mK]

    return kappa[0, 0]
