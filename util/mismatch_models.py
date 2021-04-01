"""
Developed by S. Aria Hosseini
Department of mechanical engineering
University of California, Riverside
"""

import numpy as np
from scipy import interpolate
from .wavepacket_simulator import *


def acoustic_mismatch(speed_of_sound, mass_density, n_mu=2e4):

    """
    acoustic_mismatch returns transmission coefficient of two solids in contact.
    The transmission material (i) to material (j).
    this function assumed single Debye phonon band approximation and an effective bulk stiffness

    :arg
        speed_of_sound    : Material speed of sound, An array of 1 by 2
        mass_density      : Materials mass density, An array of 1 by 2
        n_mu              : Angular sampling

    :returns
        mu_critical     : Critical angle, float number
        tij             : Angel dependent transmission coefficient from i to j, np.array
        Tij             : Angel independent transmission coefficient from i to j, np.array
        chi             : Suppression function to include angel dependency of transmission coefficient in
                          interfacial thermal conductance
        Chi             : Angel-averaged suppression function
    """

    z_i = mass_density[0] * speed_of_sound[0]       # Acoustic impedance
    z_j = mass_density[1] * speed_of_sound[1]       # Acoustic impedance

    if speed_of_sound[1] < speed_of_sound[0]:       # Transmission from stiff material to soft material
        mu_critical = 0                             # Critical angle
    else:       # Transmission from soft material to stiff material
        mu_critical = np.cos(np.arcsin(speed_of_sound[0] / speed_of_sound[1]))      # Critical angel

    mu_i = np.linspace(mu_critical, 1, int(n_mu))           # Sample incoming angele
    mu_j = np.sqrt(1 - ((speed_of_sound[1] / speed_of_sound[0]) ** 2) *
                   (1 - np.power(mu_i, 2)))                 # Sample outgoing angel

    tij = 4 * (z_i * z_j) * np.divide(np.multiply(mu_i, mu_j),
                                      np.power(z_i * mu_i + z_j * mu_j, 2))  # Angel dependent transmission coefficient

    T = np.trapz(y=tij, x=mu_i, dx=mu_i[1] - mu_i[0], axis=-1)  # Angel independent transmission coefficient
    chi = ((1 + z_j / z_i)**2 / (1 + z_j / z_i * mu_j / mu_i)**2) * (mu_j / mu_i)   # Suppression function
    Chi = np.trapz(y=chi, x=mu_i, dx=mu_i[1] - mu_i[0], axis=-1)  # Angel-averaged suppression function

    return mu_i, mu_j, mu_critical, tij, T, chi, Chi


def diffuse_mismatch(path_to_mass_weighted_hessian, speed_of_sound, eps, nq, n_sampleing):

    """
    diffuse_mismatch returns transmission coefficient of two solids in contact.
    This function assumed effective stiffness

    :arg
        path_to_mass_weighted_hessian    : Point to mass weighted hessian file, 1 by 2 list of string
        speed_of_sound                   : Material speed of sound, An array of 1 by 2
        eps                              : DoS Gaussian width to mimic delta function, floating number
        nq                               : qpoint sampling, integer
        n_sampleing                      : Frequency sampling, integer


    :returns
        tij             : Angel dependent transmission coefficient from i to j, np.array
        omg             : Phonon angular frequency

    """

    # Call vibrational_density_state method from wavepacket_simulator.py
    vibrationa_properties_i = vibrational_density_state(path_to_mass_weighted_hessian=path_to_mass_weighted_hessian[0],
                                                        eps=eps[0], nq=nq[0])
    vibrationa_properties_j = vibrational_density_state(path_to_mass_weighted_hessian=path_to_mass_weighted_hessian[1],
                                                        eps=eps[1], nq=nq[1])

    # Maximum overlap frequency
    omg_cut = np.min([np.max(vibrationa_properties_i[-1]), np.max(vibrationa_properties_i[-1])])
    omg = np.linspace(0, omg_cut, int(n_sampleing))     # Frequency

    # Interpolations
    f_i = interpolate.PchipInterpolator(vibrationa_properties_i[-1][0], vibrationa_properties_i[2][0], extrapolate=None)
    f_j = interpolate.PchipInterpolator(vibrationa_properties_j[-1][0], vibrationa_properties_j[2][0], extrapolate=None)
    
    dos_i = f_i(omg)    # Density of state in material i
    dos_j = f_j(omg)    # Density of state in material j
    tij = np.divide(speed_of_sound[1] * dos_j, speed_of_sound[0] * dos_i + speed_of_sound[1] * dos_j,
                    out=np.zeros_like(speed_of_sound[1] * dos_j),
                    where=speed_of_sound[0] * dos_i + speed_of_sound[1] * dos_j != 0)   # Transmission coefficient
    return tij, omg
