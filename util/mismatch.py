"""
Developed by S. Aria Hosseini
Department of mechanical engineering
University of California, Riverside
"""
import numpy as np
from scipy import interpolate
from .func import *


def acoustic_mismatch(speed_of_sound, mass_density, n_mu=2e4):
    """
    acoustic_mismatch returns transmission coefficient of two solids in contact.
    The transmission is always from softer material (i) to stiffer one (j).
    this function assumed Debye approximation and effective stiffness

    :arg
        omega_cutoff    : maximum frequency of the softer material
        omega_max       : maximum frequency of material i
        n_omg           : frequency sampling
        n_mu            : angular sampling

    :type
        omega_cutoff    : float number
        omega_max       : float number
        n_omg           : integer number
        n_mu            : integer number

    :returns
        mu_critical     : critical angle, float number
        tij             : angle dependent transmission coefficient from i to j, np.array
        Tij             : transmission coefficient from i to j, np.array
        MTij            : matrix of transmission coefficient, row: freq dependency
                                                column: angle dependency, np.array
    """
    z_i = mass_density[0] * speed_of_sound[0]
    z_j = mass_density[1] * speed_of_sound[1]

    if speed_of_sound[1] < speed_of_sound[0]:
        mu_critical = 0
    else:
        mu_critical = np.cos(np.arcsin(speed_of_sound[0] / speed_of_sound[1]))

    mu_i = np.linspace(mu_critical, 1, int(n_mu))
    mu_j = np.sqrt(1 - ((speed_of_sound[1] / speed_of_sound[0]) ** 2) *
                   (1 - np.power(mu_i, 2)))

    tij = 4 * (z_i * z_j) * np.divide(np.multiply(mu_i, mu_j),
                                      np.power(z_i * mu_i + z_j * mu_j, 2))

    T = np.trapz(y=tij, x=mu_i, dx=mu_i[1] - mu_i[0], axis=-1)
    chi = ((1 + z_j / z_i)**2 / (1 + z_j / z_i * mu_j / mu_i)**2) * (mu_j / mu_i)
    Chi = np.trapz(y=chi, x=mu_i, dx=mu_i[1] - mu_i[0], axis=-1)

    return mu_i, mu_j, mu_critical, tij, T, chi, Chi


def diffuse_mismatch(path_to_mass_weighted_hessian, speed_of_sound, eps, nq, n_sampleing):
    """
    diffuse_mismatch returns transmission coefficient of two solids in contact.
    The transmission is always from softer material (i) to stiffer one (j).
    this function assumed Debye approximation and effective stiffness

    :arg
        omega_cutoff    : maximum frequency of the softer material
        omega_max       : maximum frequency of material i
        n_omg           : frequency sampling
        n_mu            : angular sampling

    :type
        omega_cutoff    : float number
        omega_max       : float number
        n_omg           : integer number
        n_mu            : integer number

    :returns
        mu_critical     : critical angle, float number
        tij             : angle independent transmission coefficient from i to j, np.array
        Tij             : transmission coefficient from i to j, np.array
        MTij            : matrix of transmission coefficient, row: freq dependency
                                                column: angle dependency, np.array
    """

    vibrationa_properties_i = vibrational_density_state(path_to_mass_weighted_hessian=path_to_mass_weighted_hessian[0],
                                                        eps=eps[0], nq=nq[0])
    vibrationa_properties_j = vibrational_density_state(path_to_mass_weighted_hessian=path_to_mass_weighted_hessian[1],
                                                        eps=eps[1], nq=nq[1])
    omg_cut = np.min([np.max(vibrationa_properties_i[-1]), np.max(vibrationa_properties_i[-1])])
    omg = np.linspace(0, omg_cut, int(n_sampleing))
    f_i = interpolate.PchipInterpolator(vibrationa_properties_i[-1][0], vibrationa_properties_i[2][0], extrapolate=None)
    f_j = interpolate.PchipInterpolator(vibrationa_properties_j[-1][0], vibrationa_properties_j[2][0], extrapolate=None)
    dos_i = f_i(omg)
    dos_j = f_j(omg)
    tij = np.divide(speed_of_sound[1] * dos_j, speed_of_sound[0] * dos_i + speed_of_sound[1] * dos_j,
                    out=np.zeros_like(speed_of_sound[1] * dos_j),
                    where=speed_of_sound[0] * dos_i + speed_of_sound[1] * dos_j != 0)
    return tij, omg
