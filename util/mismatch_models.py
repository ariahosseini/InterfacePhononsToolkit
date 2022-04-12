"""
A set of methods to characterize the interfacial thermal conductivity using mismatch models
"""

import numpy as np
from scipy.interpolate import PchipInterpolator as interpolator
from .dynam import vibrational_density_state as vib_dos


def acoustic_mismatch(speed_of_sound: np.ndarray, mass_density: np.ndarray, n_mu: int = 2e4) -> dict:

    """
    Acoustic Mismatch Model (AMM).

    This function assumed single Debye phonon band approximation and an effective bulk stiffness.

    :arg
        speed_of_sound: np.ndarray
            Material speed of sound, size: (1, 2)
        mass_density: np.ndarray
            Materials mass density, size: (1, 2)
        n_mu: int
            Angular space mesh size
    :returns
        output: dict
        output['mu_critical']           : Critical angle, float number
        output['transmission_coeff']    : Angle-dependent transmission coefficient from i to j
        output['angle_independent_transmission_coeff']  : Angle-independent transmission coefficient from i to j
        output['suppression']           : Angle-averaged suppression function
    """

    z_i, z_j = mass_density * speed_of_sound  # Acoustic impedance

    if speed_of_sound[1] < speed_of_sound[0]:  # The transmission is from the stiff material to the soft material

        mu_critical = 0  # Critical angle

    else:  # The transmission is from the soft material to the stiff material

        mu_critical = np.cos(np.arcsin(speed_of_sound[0] / speed_of_sound[1]))  # Critical angle

    mu_i = np.linspace(mu_critical, 1, int(n_mu))  # Sampling the incoming angle

    mu_j = np.sqrt(1 - ((speed_of_sound[1] / speed_of_sound[0]) ** 2) * (1 - mu_i**2))  # Sampling the outgoing angle

    trans_ij = 4 * (z_i * z_j) * (mu_i*mu_j)/(z_i * mu_i + z_j * mu_j) ** 2  # Angle-dependent transmission coefficient
    trans = np.trapz(y=trans_ij, x=mu_i, dx=mu_i[1] - mu_i[0], axis=-1)  # Angle-independent transmission coefficient
    sup_func = ((1 + z_j / z_i) ** 2 / (1 + z_j / z_i * mu_j / mu_i) ** 2) * (mu_j / mu_i)  # Suppression function
    suppression = np.trapz(y=sup_func, x=mu_i, dx=mu_i[1] - mu_i[0], axis=-1)  # Angle-averaged suppression function

    output = {'mu_critical': mu_critical, 'transmission_coeff': trans_ij,
              'angle_independent_transmission_coeff': trans, 'suppression': suppression}

    return output


def diffuse_mismatch(path_to_mass_weighted_hessian: list, speed_of_sound: np.nddarray,
                     eps_o: list, num_qpoints: list, frq_sampling: int) -> list:

    """

    :arg
        path_to_mass_weighted_hessian: list['str', 'str']
            Point to mass weighted hessian files
        speed_of_sound: np.ndarray
            Materials speed of sound
        eps_o: list['str', 'str']
            DoS Gaussian width to approximate the delta function
        num_qpoints: list['str', 'str']
            Wave vectors sampling, integer
        frq_sampling: int
            Frequency sampling size
    :returns
    output: list
        trs_ij: np.ndarray
            Angle-dependent transmission coefficient from i to j
        omg: np.ndarray
            Phonon angular frequency
    """

    vib_prop_i = vib_dos(path_to_mass_weighted_hessian=path_to_mass_weighted_hessian[0],
                                                        eps=eps_o[0], nq=num_qpoints[0])
    vib_prop_j = vib_dos(path_to_mass_weighted_hessian=path_to_mass_weighted_hessian[1],
                                                        eps=eps_o[1], nq=num_qpoints[1])

    omg_cut = np.min([np.max(vib_prop_i[-1]), np.max(vib_prop_j[-1])])
    omg = np.linspace(0, omg_cut, int(frq_sampling))

    frq_i = interpolator(vib_prop_i[-1][0], vib_prop_i[2][0], extrapolate=None)
    frq_j = interpolator(vib_prop_j[-1][0], vib_prop_j[2][0], extrapolate=None)

    dos_i = frq_i(omg)  # Density of state in material i
    dos_j = frq_j(omg)  # Density of state in material j

    trs_ij = np.divide(speed_of_sound[1] * dos_j, speed_of_sound[0] * dos_i + speed_of_sound[1] * dos_j,
                    out=np.zeros_like(speed_of_sound[1] * dos_j),
                    where=speed_of_sound[0] * dos_i + speed_of_sound[1] * dos_j != 0)  # Transmission coefficient
    output = list(zip(omg, trs_ij))

    return output
