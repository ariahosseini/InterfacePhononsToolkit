"""
The is collection of functions to generate Gaussian wavepacket and the Hessian matrix
Author: S. Aria Hosseini
Email: shoss008@ucr.edu

"""

import numpy as np
import os


def vibrational_density_state(path_to_mass_weighted_hessian, eps=3e12, nq=2e4):
    """

    This function calculate vibrational density of state from hessian matrix
    of molecular dynamics calculations

    :arg
        path_to_mass_weighted_hessian       : Point to the mass weighted hessian file, string
        eps                                 : Tuning parameter, show the life time relate to line width, small eps leads
                                              to noisy vDoS and large eps leads to unrealistic vDoS, float number
        nq                                  : Sampling grid, integer number

    :returns
        hessian_matrix                      : Hessian matrix, np.array, 3N by 3N array where N is number of atoms
        frq                                 : Frequency in rad/S, can be converted to THz using conversion_factor_to_THz
        density_state                       : Vibrational density of state, an array of 1 by nq
        omg                                 : Frequency sampling corresponds to "density_state", an array of 1 by nq

    """

    with open(os.path.expanduser(path_to_mass_weighted_hessian)) as hessian_file:
        _hessian = np.loadtxt(hessian_file, delimiter=None)
    hessian_file.close()
    hessian_symmetry = (np.triu(_hessian) + np.tril(_hessian).transpose()) / 2      # Hessian matrix is Hermitian
    hessian_matrix = hessian_symmetry + np.triu(hessian_symmetry, 1).transpose()
    egn_value, egn_vector = np.linalg.eigh(hessian_matrix)                          # Note, eign_values are negative
    egn_value = np.where(egn_value < 0, egn_value, 0.0)                             # Get rid of unstable modes
    frq = np.sqrt(-1 * egn_value)                                                   # Frequency from Hessian matrix
    frq = np.sort(frq)
    frq = frq[np.newaxis]
    omg = np.linspace(np.min(frq), np.max(frq), int(nq))[np.newaxis]                # Sampling the frequency
    density_state = 1 / nq * np.sum(1 / np.sqrt(np.pi) / eps * np.exp(-1 * np.power(omg.T - frq, 2) / eps / eps),
                                    axis=1)[np.newaxis]                             # Density of state

    return hessian_matrix, frq, density_state, omg


def atoms_position(path_to_atoms_positions, num_atoms, num_atoms_unit_cell, reference_unit_cell, skip_lines=16):
    """

    This function returns the position of atoms, lattice points and positional vectors respect to a reference lattice
    point. Note that atoms are sort by their index.

    :arg
        path_to_atoms_positions             : Point to the data file in LAMMPS format, string
        num_atoms                           : Number of atoms, integer
        num_atoms_unit_cell                 : Number of atoms per unitcell
        reference_unit_cell                 : The index of the unitcell that is set as the origin (0, 0, 0),
                                              integer

    :returns
        position_of_atoms                   : Atoms position, [index, atom type, x, y, z], N by 5 array
        lattice_points                      : Lattice point [x, y, z] N/Number of atoms per unitcell by 3 array
        lattice_points_vectors              : Positional vector from origin to the unitcells,
                                              N/Number of atoms per unitcell by 3 array
    """

    with open(os.path.expanduser(path_to_atoms_positions)) as atomsPositionsFile:
        _position = np.loadtxt(atomsPositionsFile, skiprows=skip_lines, max_rows=num_atoms)
    position_of_atoms = _position[np.argsort(_position[:, 0])]
    lattice_points = position_of_atoms[::num_atoms_unit_cell, 2:5]
    lattice_points_vectors = lattice_points - lattice_points[reference_unit_cell - 1, :][np.newaxis]

    return position_of_atoms, lattice_points, lattice_points_vectors


def qpoints(num_qpoints, lattice_parameter, BZ_path):
    """
    This function samples the BZ path

    :arg
        num_qpoints                         : Sampling number, integer
        lattice_parameter                   : Lattice parameter, floating
        BZ_path                             : 1 by 3 list of [1s or 0s], where 1 is True and 0 is False

    :returns
        points                              : wave vectors along BZ_path
    """

    points = np.asarray(BZ_path)[np.newaxis].T * np.linspace(0, 2 * np.pi / lattice_parameter, num_qpoints)[np.newaxis]
    return points


def dynamical_matrix(path_to_mass_weighted_hessian, path_to_atoms_positions, num_atoms, num_atoms_unit_cell,
                     central_unit_cell, lattice_parameter, BZ_path=None, skip_lines=16, num_qpoints=1000):
    """

    This function calculate vibrational eigenvectors and frequencies from dynamical matrix
    of molecular dynamics calculations

    :arg
        path_to_mass_weighted_hessian       : Point to the mass weighted hessian file, string
        path_to_atoms_positions             : Point to the data file in LAMMPS format, string
        num_atoms                           : Number of atoms, integer
        num_atoms_unit_cell                 : Number of atoms per unitcell
        central_unit_cell                   : The index of the unitcell that is considered as the origin (0, 0, 0),
                                              It is better to be at the center and number of cells should be odd number
                                              , integer
        lattice_parameter                   : Lattice parameter, floating
        BZ_path                             : 1 by 3 list of [1s or 0s], where 1 is yes and 0 is no
        num_qpoints                         : Sampling number, integer

    :returns
        eig_vector                          : Eigenvector
        frequency                           : Frequency in rad/S, can be converted to THz using conversion_factor_to_THz
        points                              : BZ path sampling, 3 by num_qpoints array

    """

    if BZ_path is None:
        BZ_path = [0, 0, 1]

    hessian_matrix = vibrational_density_state(path_to_mass_weighted_hessian=path_to_mass_weighted_hessian)[0]
    crystal_points = atoms_position(path_to_atoms_positions=path_to_atoms_positions, num_atoms=num_atoms,
                                    num_atoms_unit_cell=num_atoms_unit_cell, reference_unit_cell=central_unit_cell,
                                    skip_lines=skip_lines)
    dynamical_matrix = np.zeros((num_atoms_unit_cell * 3, num_atoms_unit_cell * 3))
    points = qpoints(num_qpoints=num_qpoints, lattice_parameter=lattice_parameter, BZ_path=BZ_path)
    for _ in range(num_qpoints):
        dynamical_matrix_per_qpoint = np.zeros((num_atoms_unit_cell * 3, num_atoms_unit_cell * 3))
        for __ in range(len(crystal_points[2])):
            sum_matrix = hessian_matrix[__ * num_atoms_unit_cell * 3: (__ + 1) * num_atoms_unit_cell * 3,
                                        (central_unit_cell - 1) * num_atoms_unit_cell * 3:
                                        (central_unit_cell) * num_atoms_unit_cell * 3] * \
                np.exp(-1j * np.dot(crystal_points[2][__], points[:, _]))
            dynamical_matrix_per_qpoint = dynamical_matrix_per_qpoint + sum_matrix
        dynamical_matrix = np.append(dynamical_matrix, dynamical_matrix_per_qpoint, axis=0)

    dynamical_matrix = dynamical_matrix[num_atoms_unit_cell * 3:]

    eig_value = np.array([])
    eig_vector = np.zeros((num_atoms_unit_cell * 3, num_atoms_unit_cell * 3))

    for _ in range(num_qpoints):
        dynmat = dynamical_matrix[_ * num_atoms_unit_cell * 3:(_ + 1) * num_atoms_unit_cell * 3]
        eigvals, __eigvecs, = np.linalg.eigh(dynmat)
        energy = -1 * eigvals.real
        order = np.argsort(energy)
        energy = energy[order]
        _eigvecs, _ = np.linalg.qr(__eigvecs)
        eigvecs = np.array([_eigvecs[_][order] for _ in range(np.shape(_eigvecs)[1])])
        eig_value = np.append(eig_value, energy).reshape(-1, num_atoms_unit_cell * 3)
        eig_vector = np.append(eig_vector, eigvecs, axis=0)
    eig_vector = eig_vector[num_atoms_unit_cell * 3:]
    frequency = np.sqrt(np.abs(eig_value))

    return eig_vector, frequency, points


def acoustic_phonon(path_to_mass_weighted_hessian, path_to_atoms_positions, num_atoms, num_atoms_unit_cell,
                    central_unit_cell, lattice_parameter, BZ_path=None, skip_lines=16,
                    num_qpoints=1000):
    """
        This function returns transverse and longitudinal eigenmodes from dynamical matrix
        of molecular dynamics calculations

        :arg
            path_to_mass_weighted_hessian       : Point to the mass weighted hessian file, string
            path_to_atoms_positions             : Point to the data file in LAMMPS format, string
            num_atoms                           : Number of atoms, integer
            num_atoms_unit_cell                 : Number of atoms per unitcell
            central_unit_cell                   : The index of the unitcell that is considered as the origin (0, 0, 0),
                                                  It is better to be at the center and number of cells
                                                  should be odd number, integer
            lattice_parameter                   : Lattice parameter, floating
            path                                : 1 by 3 list of [1s or 0s], where 1 is True and 0 is False
            num_qpoints                         : Sampling number, integer
            intersection                        : any possible cross band. This works for one cross band between 3rd and
                                                  fifth bands. It needs modification for other cases (simple).

        :returns
            eigvector                           : Eigenvector
            eigenvalues                         : Frequency in rad/S, can be converted to THz using
                                                  conversion_factor_to_THz
            frq[2]                              : BZ path sampling, 3 by num_qpoints array
        """

    if BZ_path is None:
        BZ_path = [0, 0, 1]

    frq = dynamical_matrix(path_to_mass_weighted_hessian=path_to_mass_weighted_hessian,
                           path_to_atoms_positions=path_to_atoms_positions, num_atoms=num_atoms,
                           num_atoms_unit_cell=num_atoms_unit_cell, central_unit_cell=central_unit_cell,
                           lattice_parameter=lattice_parameter, BZ_path=BZ_path, skip_lines=skip_lines,
                           num_qpoints=num_qpoints)

    longitudinal_mode_frq = frq[1][:, 2]
    transverse_mode_frq = frq[1][:, 0]

    longitudinal_eigvec = np.reshape(frq[0][:, 2], (num_qpoints, num_atoms_unit_cell * 3)).T

    _transverse_mode_eigvec = np.reshape(frq[0][:, 0], (num_qpoints, num_atoms_unit_cell * 3))
    __transverse_mode_eigvec = np.reshape(frq[0][:, 1], (num_qpoints, num_atoms_unit_cell * 3))

    angle_x = np.array([np.arctan(np.divide(np.sum(-1 * np.multiply(_transverse_mode_eigvec,
                                                                    np.tile(np.array([0, 1, 0]), (int(num_qpoints),
                                                                                                  num_atoms_unit_cell))
                                                                    ), axis=1),
                                            np.sum(-1 * np.multiply(__transverse_mode_eigvec,
                                                                    np.tile(np.array([0, 1, 0]), (int(num_qpoints),
                                                                                                  num_atoms_unit_cell))
                                                                    ), axis=1)))])
    angle_y = np.array([np.arctan(np.divide(np.sum(-1 * np.multiply(_transverse_mode_eigvec,
                                                                    np.tile(np.array([1, 0, 0]), (int(num_qpoints),
                                                                                                  num_atoms_unit_cell))
                                                                    ), axis=1),
                                            np.sum(-1 * np.multiply(__transverse_mode_eigvec,
                                                                    np.tile(np.array([1, 0, 0]), (int(num_qpoints),
                                                                                                  num_atoms_unit_cell))
                                                                    ), axis=1)))])

    transverse_eigvec_x = np.multiply(_transverse_mode_eigvec,
                                      np.tile(np.cos(angle_x).T, (1, 3 * num_atoms_unit_cell))) +\
        np.multiply(__transverse_mode_eigvec,
                    np.tile(np.sin(angle_x).T, (1, 3 * num_atoms_unit_cell)))

    transverse_eigvec_y = np.multiply(_transverse_mode_eigvec,
                                      np.tile(np.cos(angle_y).T, (1, 3 * num_atoms_unit_cell))) +\
        np.multiply(__transverse_mode_eigvec,
                    np.tile(np.sin(angle_y).T, (1, 3 * num_atoms_unit_cell)))

    eigenvalue = np.array([transverse_mode_frq, transverse_mode_frq, longitudinal_mode_frq])
    eigenvector = np.array([transverse_eigvec_x.T, transverse_eigvec_y.T, longitudinal_eigvec])

    return eigenvalue, eigenvector, frq[2]


def gaussian_distribution(amplitude, sigma, wavenumber_idx, num_qpoints, lattice_parameter, BZ_path=None):
    """

    This function find the Gaussian distribution around "wavenumber_idx" with variance of "sigma" and amplitude of
    "amplitude"

    :arg
        amplitude                         : Amplitude of the Gaussian, floating number
        sigma                             : Broadening factor, floating
        wavenumber_idx                    : Gaussian mean value, floating
        num_qpoints                       : Sampling size, integer
        lattice_parameter                 : Lattice parameter, floating
        BZ_path                           : 1 by 3 list of [1s or 0s], where 1 is yes and 0 is no

    :returns
        gaussian[np.newaxis]              : Gaussian, 1 by num_qpoints array
        points_norm[np.newaxis]           : wavevectors, 1 by num_qpoints array

    """

    if BZ_path is None:
        BZ_path = [0, 0, 1]

    points = qpoints(num_qpoints=num_qpoints, lattice_parameter=lattice_parameter, BZ_path=BZ_path)
    points_norm = np.linalg.norm(points, axis=0)
    expected_value = points_norm[wavenumber_idx]
    gaussian = amplitude * (1.0 / np.sqrt(2 * np.pi) / sigma) * np.exp(
        -1.0 * np.power(((points_norm - expected_value) / np.sqrt(2) / sigma), 2))

    return gaussian[np.newaxis], points_norm[np.newaxis]


def single_wave(path_to_mass_weighted_hessian, path_to_atoms_positions, num_atoms, num_atoms_unit_cell,
                central_unit_cell, lattice_parameter, frq_mode, idx_ko, amplitude, sigma, replication,
                BZ_path=None, origin_unit_cell=0, skip_lines=[9, 9], num_qpoints=1000):
    """

    This function generates a single wave

    :arg
        path_to_mass_weighted_hessian       : Point to the mass weighted hessian file, string
        path_to_atoms_positions             : Point to the data file in LAMMPS format for Hessian and wave packet,
                                              list of strings
        num_atoms                           : Number of atoms for Hessian and wave packet, list of integers
        num_atoms_unit_cell                 : Number of atoms per unitcell
        central_unit_cell                   : The index of the unitcell that is considered as the origin (0, 0, 0),
                                              It is better to be at the center and number of
                                              cells should be an odd number, integer
        lattice_parameter                   : Lattice parameter, floating
        frq_mode                            : Frequency mode can be 0, 1 , 2, for longitudinal , transverses modes
        idx_ko                              : Gaussian center, integer
        amplitude                           : Amplitude of the Gaussian, floating number
        sigma                               : Broadening factor, floating
        replication                         : Size of the box in unit of unitcell, [x, y, z]
        BZ_path                             : 1 by 3 list of [1s or 0s], where 1 is yes and 0 is no
        origin_unit_cell                    : Reference unitcell, 0 is reasonable
        skip_lines                          : Skip headers
        num_qpoints                         : Sampling size, integer


    :returns
        positions[0]                        : Position of atoms
        frq[0]                              : Frequency
        gaussian                            : Gaussian distribution
        points                              : qpoints
        solid_lattice_points                : Lattice points
        wave.real                           : Wave

    """

    if BZ_path is None:
        BZ_path = [0, 0, 1]

    num_cell = replication[0] * replication[1] * replication[2]
    positions = atoms_position(path_to_atoms_positions=path_to_atoms_positions[1], num_atoms=num_atoms[1],
                               num_atoms_unit_cell=num_atoms_unit_cell, reference_unit_cell=origin_unit_cell,
                               skip_lines=skip_lines[1])

    solid_lattice_points = positions[2][:num_cell]

    frq = acoustic_phonon(path_to_mass_weighted_hessian=path_to_mass_weighted_hessian,
                          path_to_atoms_positions=path_to_atoms_positions[0], num_atoms=num_atoms[0],
                          num_atoms_unit_cell=num_atoms_unit_cell, central_unit_cell=central_unit_cell,
                          lattice_parameter=lattice_parameter, BZ_path=BZ_path, skip_lines=skip_lines[0],
                          num_qpoints=num_qpoints)

    gaussian = gaussian_distribution(amplitude=amplitude, sigma=sigma, wavenumber_idx=idx_ko, num_qpoints=num_qpoints,
                                     lattice_parameter=lattice_parameter, BZ_path=BZ_path)[0]

    points = qpoints(num_qpoints=num_qpoints, lattice_parameter=lattice_parameter, BZ_path=BZ_path)

    sign_correction = np.exp(-1j * (np.arctan(frq[1][frq_mode][frq_mode, idx_ko].imag /
                                              frq[1][frq_mode][frq_mode, idx_ko].real)))

    wv_x = gaussian[0, idx_ko] * \
        np.multiply(np.tile(frq[1][frq_mode][::3, idx_ko][np.newaxis].T, (num_cell, 1)),
                    np.exp(-1j * np.matmul(np.reshape(np.tile(solid_lattice_points,
                                                              num_atoms_unit_cell),
                                                      (num_cell * num_atoms_unit_cell, 3)),
                                           points[:, idx_ko][np.newaxis].T))) * sign_correction
    wv_y = gaussian[0, idx_ko] * \
        np.multiply(np.tile(frq[1][frq_mode][1::3, idx_ko][np.newaxis].T, (num_cell, 1)),
                    np.exp(-1j * np.matmul(np.reshape(np.tile(solid_lattice_points,
                                                              num_atoms_unit_cell),
                                                      (num_cell * num_atoms_unit_cell, 3)),
                                           points[:, idx_ko][np.newaxis].T))) * sign_correction
    wv_z = gaussian[0, idx_ko] * \
        np.multiply(np.tile(frq[1][frq_mode][2::3, idx_ko][np.newaxis].T, (num_cell, 1)),
                    np.exp(-1j * np.matmul(np.reshape(np.tile(solid_lattice_points,
                                                              num_atoms_unit_cell),
                                                      (num_cell * num_atoms_unit_cell, 3)),
                                           points[:, idx_ko][np.newaxis].T))) * sign_correction

    wave = np.concatenate((wv_x.T, wv_y.T, wv_z.T), axis=0)

    return positions[0], frq[0], gaussian, points, solid_lattice_points, wave.real


def wavepacket(path_to_mass_weighted_hessian, path_to_atoms_positions, num_atoms,
               num_atoms_unit_cell, central_unit_cell, amplitude, lattice_parameter, frq_mode,
               idx_ko, sigma, replication, time=10, BZ_path=None, origin_unit_cell=0, skip_lines=[9, 9],
               num_qpoints=500):
    """

    This function generates a wavepacket

    :arg
        path_to_mass_weighted_hessian       : Point to the mass weighted hessian file, string
        path_to_atoms_positions             : Point to the data file in LAMMPS format for Hessian and wave packet,
                                              list of strings
        num_atoms                           : Number of atoms for Hessian and wave packet, list of integers
        num_atoms_unit_cell                 : Number of atoms per unitcell
        central_unit_cell                   : The index of the unitcell that is considered as the origin (0, 0, 0),
                                              It is better to be at the center and number of
                                              cells should be an odd number, integer
        lattice_parameter                   : Lattice parameter, floating
        frq_mode                            : Frequency mode can be 0, 1 , 2, for longitudinal , transverses modes
        idx_ko                              : Gaussian center, integer
        amplitude                           : Amplitude of the Gaussian, floating number
        sigma                               : Broadening factor, floating
        replication                         : Size of the box in unit of unitcell, [x, y, z]
        time                                : Time, it can be zero without loss of generality
        BZ_path                             : 1 by 3 list of [1s or 0s], where 1 is yes and 0 is no
        origin_unit_cell                    : Reference unitcell, 0 is reasonable
        skip_lines                          : Skip headers
        num_qpoints                         : Sampling size, integer


    :returns
        positions[0]                        : Position of atoms
        phonon_wavepacket                   : Phonon Wavepacket

    """

    if BZ_path is None:
        BZ_path = [0, 0, 1]

    num_cell = replication[0] * replication[1] * replication[2]
    positions = atoms_position(path_to_atoms_positions=path_to_atoms_positions[1], num_atoms=num_atoms[1],
                               num_atoms_unit_cell=num_atoms_unit_cell, reference_unit_cell=origin_unit_cell,
                               skip_lines=skip_lines[1])

    solid_lattice_points = positions[2][:num_cell]

    frq = acoustic_phonon(path_to_mass_weighted_hessian=path_to_mass_weighted_hessian,
                          path_to_atoms_positions=path_to_atoms_positions[0], num_atoms=num_atoms[0],
                          num_atoms_unit_cell=num_atoms_unit_cell, central_unit_cell=central_unit_cell,
                          lattice_parameter=lattice_parameter, BZ_path=BZ_path, skip_lines=skip_lines[0],
                          num_qpoints=num_qpoints)

    gaussian = gaussian_distribution(amplitude=amplitude, sigma=sigma, wavenumber_idx=idx_ko, num_qpoints=num_qpoints,
                                     lattice_parameter=lattice_parameter, BZ_path=BZ_path)[0]

    points = qpoints(num_qpoints=num_qpoints, lattice_parameter=lattice_parameter, BZ_path=BZ_path)

    sign_correction = np.exp(-1j * (np.arctan(np.divide(frq[1][frq_mode][frq_mode, :].imag[np.newaxis],
                                                        frq[1][frq_mode][frq_mode, :].real[np.newaxis]))))

    omega_time = np.exp(1j * (2 * np.pi) * time * frq[0][frq_mode, :])[np.newaxis]

    wv_x = np.sum(np.multiply(np.multiply(np.tile(np.multiply(np.multiply(gaussian, sign_correction),
                                                              omega_time), (num_cell * num_atoms_unit_cell, 1)),
                                          np.tile(frq[1][frq_mode][::3, :], (num_cell, 1))),
                              np.exp(-1j * np.matmul(np.reshape(np.tile(solid_lattice_points, num_atoms_unit_cell),
                                                                (num_cell * num_atoms_unit_cell, 3)), points))),
                  axis=1)[np.newaxis]

    wv_y = np.sum(np.multiply(np.multiply(np.tile(np.multiply(np.multiply(gaussian, sign_correction),
                                                              omega_time), (num_cell * num_atoms_unit_cell, 1)),
                                          np.tile(frq[1][frq_mode][1::3, :], (num_cell, 1))),
                              np.exp(-1j * np.matmul(np.reshape(np.tile(solid_lattice_points, num_atoms_unit_cell),
                                                                (num_cell * num_atoms_unit_cell, 3)), points))),
                  axis=1)[np.newaxis]

    wv_z = np.sum(np.multiply(np.multiply(np.tile(np.multiply(np.multiply(gaussian, sign_correction),
                                                              omega_time), (num_cell * num_atoms_unit_cell, 1)),
                                          np.tile(frq[1][frq_mode][2::3, :], (num_cell, 1))),
                              np.exp(-1j * np.matmul(np.reshape(np.tile(solid_lattice_points, num_atoms_unit_cell),
                                                                (num_cell * num_atoms_unit_cell, 3)), points))),
                  axis=1)[np.newaxis]

    phonon_wavepacket = np.concatenate((wv_x, wv_y, wv_z), axis=0)

    return phonon_wavepacket, positions[0]
