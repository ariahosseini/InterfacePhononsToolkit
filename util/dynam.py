"""
The is collection of methods to generate phonon wave-packets
"""

import numpy as np


def vibrational_density_state(path_to_mass_weighted_hessian: str, eps_o: float = 3e12, nq: int = 2e4):

    """
    Compute the vibrational density of state from hessian matrix

    :arg
        path_to_mass_weighted_hessian: str
               Point to the mass weighted hessian file
        eps_o: float
            The bandwidth — small eps leads to noisy vDoS and large eps leads to unrealistic vDoS
        nq: int
            Sampling grid size
    :returns
        hessian_matrix: np.ndarray
            Hessian matrix
        frq: np.ndarray
            Frequency [rad/sec]
        density_state: np.ndarray
            Vibrational density of state
        omg: np.ndarray
            Frequency sampling of "density_state"
    """

    _hessian = np.loadtxt(path_to_mass_weighted_hessian, delimiter=None)
    hessian_symmetry = (np.triu(_hessian) + np.tril(_hessian).T) / 2
    hessian_matrix = hessian_symmetry + np.triu(hessian_symmetry, 1).T
    egn_value, egn_vector = np.linalg.eigh(hessian_matrix)  # Note that egn_value is negative
    egn_value = np.where(egn_value < 0, egn_value, 0.0)  # Get rid of unstable modes
    _frq = np.sqrt(-1 * egn_value)  # Frequency of Hessian matrix
    frq = np.sort(_frq)[np.newaxis]
    omg = np.linspace(np.min(frq), np.max(frq), int(nq))[np.newaxis]  # Sampling the frequency
    density_state = 1 / nq * np.sum(1 / np.sqrt(np.pi) / eps_o * np.exp(-1 * np.power(omg.T - frq, 2) / eps_o / eps_o),
                                    axis=1)[np.newaxis]  # Density of states

    return hessian_matrix, frq, density_state, omg


def atoms_position(path_to_atoms_positions: str, num_atoms: int, num_atoms_unit_cell: int,
                   reference_unit_cell: int, skip_lines: int = 16):

    """
    Returns the position of atoms, lattice points and positional vectors respect to a reference lattice
    point. Note that atoms are sort by their index.

    :arg
        path_to_atoms_positions: str
                Point to the data file
        num_atoms: int
                Number of atoms
        num_atoms_unit_cell: int
                Number of atoms per unitcell
        reference_unit_cell: int
                The index of the unit cell that is set as the origin
    :returns
        position_atoms: np.ndarray
                Atoms' position, [index, atom type, x, y, z]
        lattice_points: np.ndarray
                Lattice point
        lattice_points_vectors: np.ndarray
                Positional vector from origin to the unit cells,
    """

    position = np.loadtxt(path_to_atoms_positions, skiprows=skip_lines, max_rows=num_atoms)

    position_atoms = position[np.argsort(position[:, 0])]
    lattice_points = position_atoms[::num_atoms_unit_cell, 2:5]
    lattice_points_vectors = lattice_points - lattice_points[reference_unit_cell - 1, :][np.newaxis]

    return position_atoms, lattice_points, lattice_points_vectors


def qpoints(num_qpoints: int, lattice_parameter: float, BZ_path: list) -> np.ndarray:

    """
    This function samples the BZ path

    :arg
        num_qpoints: int
            Sampling size
        lattice_parameter: float
            Lattice parameter
        BZ_path: list
            A list of (1, 3) of 1s or 0s — 1 is True and 0 is False
    :returns
        points: np.ndarray
            Wave-vectors along BZ_path
    """

    points = np.array(BZ_path)[np.newaxis].T * np.linspace(0, 2 * np.pi / lattice_parameter, num_qpoints)[np.newaxis]

    return points


def dynamical_matrix(path_to_mass_weighted_hessian: str, path_to_atoms_positions: str,
                     num_atoms: int, num_atoms_unit_cell: int, central_unit_cell: int,
                     lattice_parameter: float, BZ_path: list = None, skip_lines: int = 16,
                     num_qpoints: int = 1000):

    """
    A method to calculate vibrational eigenvectors and frequencies from dynamical matrix

    :arg
        path_to_mass_weighted_hessian: str
               Point to the mass weighted hessian file
        path_to_atoms_positions: str
               Point to the data file in LAMMPS format
        num_atoms: int
               Number of atoms
        num_atoms_unit_cell: int
                Number of atoms per unitcell
        central_unit_cell: int
                The index of the unitcell that is considered as the origin (0, 0, 0),
                It is better to be at the center and number of cells should be an odd number
        lattice_parameter: float
                Lattice parameter
        BZ_path: list
                A 1 by 3 list of [1s or 0s], where 1 is true and 0 is false
        num_qpoints: int
            Sampling size
    :returns
        eig_vector: np.ndarray
            Eigenvector
        frequency: np.ndarray
            Frequency [rad/sec]
        points: np.ndarray
            BZ path sampling
    """

    if BZ_path is None:
        BZ_path = [0, 0, 1]

    hessian_matrix = vibrational_density_state(path_to_mass_weighted_hessian=path_to_mass_weighted_hessian)[0]
    crystal_points = atoms_position(path_to_atoms_positions=path_to_atoms_positions, num_atoms=num_atoms,
                                    num_atoms_unit_cell=num_atoms_unit_cell, reference_unit_cell=central_unit_cell,
                                    skip_lines=skip_lines)
    dynamical_mat = np.zeros((num_atoms_unit_cell * 3, num_atoms_unit_cell * 3))
    points = qpoints(num_qpoints=num_qpoints, lattice_parameter=lattice_parameter, BZ_path=BZ_path)
    for _ in range(num_qpoints):
        dynamical_matrix_per_qpoint = np.zeros((num_atoms_unit_cell * 3, num_atoms_unit_cell * 3))
        for __ in range(len(crystal_points[2])):
            sum_matrix = hessian_matrix[__ * num_atoms_unit_cell * 3: (__ + 1) * num_atoms_unit_cell * 3,
                                        (central_unit_cell - 1) * num_atoms_unit_cell * 3:
                                        (central_unit_cell) * num_atoms_unit_cell * 3] * \
                np.exp(-1j * np.dot(crystal_points[2][__], points[:, _]))
            dynamical_matrix_per_qpoint += sum_matrix
        dynamical_mat = np.append(dynamical_mat, dynamical_matrix_per_qpoint, axis=0)

    dynam = dynamical_mat[num_atoms_unit_cell * 3:]

    eig_value = np.array([])
    eig_vector = np.zeros((num_atoms_unit_cell * 3, num_atoms_unit_cell * 3))

    for _ in range(num_qpoints):
        dyn_mat = dynam[_ * num_atoms_unit_cell * 3:(_ + 1) * num_atoms_unit_cell * 3]
        eigvals, __eigvecs, = np.linalg.eigh(dyn_mat)
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


def acoustic_phonon(path_to_mass_weighted_hessian: str, path_to_atoms_positions: str, num_atoms: int,
                    num_atoms_unit_cell: int, central_unit_cell: int, lattice_parameter: float, BZ_path: list = None,
                    skip_lines: int = 16, num_qpoints: int = 1000):

    """
    This function returns transverse and longitudinal eigen modes from dynamical matrix

    :arg
        path_to_mass_weighted_hessian: str
               Point to the mass weighted hessian file
        path_to_atoms_positions: str
               Point to the data file in LAMMPS format
        num_atoms: int
               Number of atoms
        num_atoms_unit_cell: int
                Number of atoms per unitcell
        central_unit_cell: int
                The index of the unitcell that is considered as the origin (0, 0, 0),
                It is better to be at the center and number of cells should be an odd number
        lattice_parameter: float
                Lattice parameter
        BZ_path: list
                A 1 by 3 list of [1s or 0s], where 1 is true and 0 is false
        num_qpoints: int
            Sampling size

    :returns
        eig_vector: np.ndarray
            Eigenvector
        frequency: np.ndarray
            Frequency [rad/sec]
        points: np.ndarray
            BZ path sampling
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
    points = frq[2]

    return eigenvalue, eigenvector, points


def gaussian_distribution(amplitude: float, sigma: float, wave_number_idx: int, num_qpoints: int,
                          lattice_parameter: float, BZ_path: list = None):

    """
    This function finds the Gaussian distribution around given wave vector "wave_number_idx"
    with variance of "sigma" and amplitude of "amplitude".

    :arg
        amplitude: float
            Amplitude of the Gaussian distribution
        sigma: float
            Broadening factor
        wave_number_idx: int
            Gaussian mean value
        num_qpoints: int
            Sampling size
        lattice_parameter: float
            Lattice parameter
        BZ_path: list
            A 1 by 3 list of [1s or 0s], where 1 is true and 0 is false
    :returns
        gaussian: np.ndarray
            Gaussian distribution
        points_norm: np.ndarray
            Wave vectors
    """

    if BZ_path is None:
        BZ_path = [0, 0, 1]

    points = qpoints(num_qpoints=num_qpoints, lattice_parameter=lattice_parameter, BZ_path=BZ_path)
    points_norm = np.linalg.norm(points, axis=0)
    expected_value = points_norm[wave_number_idx]
    gaussian = amplitude * (1.0 / np.sqrt(2 * np.pi) / sigma) * np.exp(
        -1.0 * np.power(((points_norm - expected_value) / np.sqrt(2) / sigma), 2))

    return gaussian[np.newaxis], points_norm[np.newaxis]


def single_wave(path_to_mass_weighted_hessian: str, path_to_atoms_positions: list, num_atoms: list,
                num_atoms_unit_cell: int, central_unit_cell: int, lattice_parameter: float, frq_mode: int,
                idx_ko: int, amplitude: float, sigma: float, replication: list, BZ_path: list = None,
                origin_unit_cell: int = 0, skip_lines: list = None, num_qpoints: int = 1000) -> dict:

    """
    A method to generate a single phonon wave

    :arg
        path_to_mass_weighted_hessian: str
               Point to the mass weighted hessian file
        path_to_atoms_positions: list
               Point to the data file for Hessian and wave packet, respectively
        num_atoms: list
               Number of atoms in Hessian and wave packet structures, respectively
        num_atoms_unit_cell: int
                Number of atoms per unitcell
        central_unit_cell: int
                The index of the unitcell that is considered as the origin (0, 0, 0),
                It is better to be at the center and number of cells should be an odd number
        lattice_parameter: float
                Lattice parameter
        frq_mode: int
                Frequency mode can be 1: 'longitudinal' , 2: 'transverses_mode_one', 3: 'transverses_mode_two'
        idx_ko: int
                Central of Gaussian — wave vector index

        amplitude: float
            Amplitude of the Gaussian distribution
        sigma: float
            Broadening factor
        wave_number_idx: int
            Gaussian mean value
        num_qpoints: int
            Sampling size
        lattice_parameter: float
            Lattice parameter
        BZ_path: list
            A 1 by 3 list of [1s or 0s], where 1 is true and 0 is false
        origin_unit_cell: int
            Reference unitcell index, 0 is reasonable
        replication: list
            Size of the box
        skip_lines: list
            Skip headers in the data file for Hessian and wave packet, respectively

    :returns
        output: dict
            output['atom_position'] — Position of atoms
            output['frequency'] — Frequency
            output['gaussian_dist'] — Gaussian distribution
            output['qpoints'] — qpoints
            output['lattice_points'] — Lattice points
            output['wave'] — The wave
    """

    if BZ_path is None:
        BZ_path = [0, 0, 1]

    if skip_lines is None:
        skip_lines = [9, 9]

    num_cell = np.prod(replication)

    positions = atoms_position(path_to_atoms_positions=path_to_atoms_positions[1], num_atoms=num_atoms[1],
                               num_atoms_unit_cell=num_atoms_unit_cell, reference_unit_cell=origin_unit_cell,
                               skip_lines=skip_lines[1])

    solid_lattice_points = positions[2][:num_cell]

    frq = acoustic_phonon(path_to_mass_weighted_hessian=path_to_mass_weighted_hessian,
                          path_to_atoms_positions=path_to_atoms_positions[0], num_atoms=num_atoms[0],
                          num_atoms_unit_cell=num_atoms_unit_cell, central_unit_cell=central_unit_cell,
                          lattice_parameter=lattice_parameter, BZ_path=BZ_path, skip_lines=skip_lines[0],
                          num_qpoints=num_qpoints)

    gaussian = gaussian_distribution(amplitude=amplitude, sigma=sigma, wave_number_idx=idx_ko, num_qpoints=num_qpoints,
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

    output = {'atom_position': positions[0], 'frequency': frq[0], 'gaussian_dist': gaussian,
              'qpoints': points, 'lattice_points': solid_lattice_points, 'wave': wave.real}

    return output


def wavepacket(path_to_mass_weighted_hessian: str, path_to_atoms_positions: list, num_atoms: list,
                num_atoms_unit_cell: int, central_unit_cell: int, amplitude: float, lattice_parameter: float,
               frq_mode: int, idx_ko: int, sigma: float, replication: list, time: float = 0, BZ_path = None,
               origin_unit_cell: int = 0, skip_lines: list = None, num_qpoints: int = 1000) -> dict:
    """
    A method to build a wavepacket

    :arg
        path_to_mass_weighted_hessian: str
               Point to the mass weighted hessian file
        path_to_atoms_positions: list
               Point to the data file for Hessian and wave packet, respectively
        num_atoms: list
               Number of atoms in Hessian and wave packet structures, respectively
        num_atoms_unit_cell: int
                Number of atoms per unitcell
        central_unit_cell: int
                The index of the unitcell that is considered as the origin (0, 0, 0),
                It is better to be at the center and number of cells should be an odd number
        lattice_parameter: float
                Lattice parameter
        frq_mode: int
                Frequency mode can be 1: 'longitudinal' , 2: 'transverses_mode_one', 3: 'transverses_mode_two'
        idx_ko: int
                Central of Gaussian — wave vector index

        amplitude: float
            Amplitude of the Gaussian distribution
        sigma: float
            Broadening factor
        wave_number_idx: int
            Gaussian mean value — the index
        time: float
            Time
        num_qpoints: int
            Sampling size
        lattice_parameter: float
            Lattice parameter
        BZ_path: list
            A 1 by 3 list of [1s or 0s], where 1 is true and 0 is false
        origin_unit_cell: int
            Reference unitcell index, 0 is reasonable
        replication: list
            Size of the box
        skip_lines: list
            Skip headers in the data file for Hessian and wave packet, respectively

    :returns
        output: dict
            output['atom_position'] — Position of atoms
            output['frequency'] — Frequency
            output['gaussian_dist'] — Gaussian distribution
            output['qpoints'] — qpoints
            output['lattice_points'] — Lattice points
            output['wavepacket'] — The wave

    :returns
        positions[0]                        : Position of atoms
        phonon_wavepacket                   : Phonon Wavepacket
    """

    if BZ_path is None:
        BZ_path = [0, 0, 1]

    if skip_lines is None:
        skip_lines = [9, 9]

    num_cell = np.prod(replication)

    positions = atoms_position(path_to_atoms_positions=path_to_atoms_positions[1], num_atoms=num_atoms[1],
                               num_atoms_unit_cell=num_atoms_unit_cell, reference_unit_cell=origin_unit_cell,
                               skip_lines=skip_lines[1])

    solid_lattice_points = positions[2][:num_cell]

    frq = acoustic_phonon(path_to_mass_weighted_hessian=path_to_mass_weighted_hessian,
                          path_to_atoms_positions=path_to_atoms_positions[0], num_atoms=num_atoms[0],
                          num_atoms_unit_cell=num_atoms_unit_cell, central_unit_cell=central_unit_cell,
                          lattice_parameter=lattice_parameter, BZ_path=BZ_path, skip_lines=skip_lines[0],
                          num_qpoints=num_qpoints)

    gaussian = gaussian_distribution(amplitude=amplitude, sigma=sigma, wave_number_idx=idx_ko, num_qpoints=num_qpoints,
                                     lattice_parameter=lattice_parameter, BZ_path=BZ_path)[0]

    points = qpoints(num_qpoints=num_qpoints, lattice_parameter=lattice_parameter, BZ_path=BZ_path)

    sign_correction = np.exp(-1j * (np.arctan(np.divide(frq[1][frq_mode][frq_mode, :].imag[np.newaxis],
                                                        frq[1][frq_mode][frq_mode, :].real[np.newaxis]))))

    plane_wave = np.exp(1j * (2 * np.pi) * time * frq[0][frq_mode, :])[np.newaxis]

    wv_x = np.sum(np.multiply(np.multiply(np.tile(np.multiply(np.multiply(gaussian, sign_correction),
                                                              plane_wave), (num_cell * num_atoms_unit_cell, 1)),
                                          np.tile(frq[1][frq_mode][::3, :], (num_cell, 1))),
                              np.exp(-1j * np.matmul(np.reshape(np.tile(solid_lattice_points, num_atoms_unit_cell),
                                                                (num_cell * num_atoms_unit_cell, 3)), points))),
                  axis=1)[np.newaxis]

    wv_y = np.sum(np.multiply(np.multiply(np.tile(np.multiply(np.multiply(gaussian, sign_correction),
                                                              plane_wave), (num_cell * num_atoms_unit_cell, 1)),
                                          np.tile(frq[1][frq_mode][1::3, :], (num_cell, 1))),
                              np.exp(-1j * np.matmul(np.reshape(np.tile(solid_lattice_points, num_atoms_unit_cell),
                                                                (num_cell * num_atoms_unit_cell, 3)), points))),
                  axis=1)[np.newaxis]

    wv_z = np.sum(np.multiply(np.multiply(np.tile(np.multiply(np.multiply(gaussian, sign_correction),
                                                              plane_wave), (num_cell * num_atoms_unit_cell, 1)),
                                          np.tile(frq[1][frq_mode][2::3, :], (num_cell, 1))),
                              np.exp(-1j * np.matmul(np.reshape(np.tile(solid_lattice_points, num_atoms_unit_cell),
                                                                (num_cell * num_atoms_unit_cell, 3)), points))),
                  axis=1)[np.newaxis]

    phonon_wavepacket = np.concatenate((wv_x, wv_y, wv_z), axis=0)

    output = {'atom_position': positions[0], 'frequency': frq[0], 'gaussian_dist': gaussian,
              'qpoints': points, 'lattice_points': solid_lattice_points, 'wavepacket': phonon_wavepacket}

    return output
