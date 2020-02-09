"""
Developed by S. Aria Hosseini
Department of mechanical engineering
University of California, Riverside
"""
import numpy as np
import numpy.matlib
from numpy import linalg as LA
from mpl_toolkits.mplot3d import Axes3D  # This import registers the 3D projection, but is otherwise unused
from scipy import interpolate
import math
import cmath
import os
from sys import exit
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

global conversion_factor_to_THz
conversion_factor_to_THz = 1 / 2 / math.pi / 1e12  # Convert rad/s to THz


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
        hm_tmp_1 = hessian_file.readlines()  # hm stands for hessian matrix
    hm_tmp_2 = [line.split() for line in hm_tmp_1]
    hm_tmp_3 = np.array([[float(_) for _ in __] for __ in hm_tmp_2])
    hessian_file.close()
    hessian_symmetry = (np.triu(hm_tmp_3) + np.tril(hm_tmp_3).transpose()) / 2  # Hessian matrix is Hermitian
    hessian_matrix = hessian_symmetry + np.triu(hessian_symmetry, 1).transpose()
    egn_value, egn_vector = np.linalg.eigh(hessian_matrix)  # egn_value are negative
    egn_value = np.where(egn_value < 0, egn_value, 0)  # Get rid of unstable modes
    frq = np.sqrt(-1 * egn_value)  # Frequency from Hessian matrix
    frq = np.sort(frq)
    frq = frq[np.newaxis]
    omg = np.linspace(np.min(frq), np.max(frq), nq)[np.newaxis]  # Sampling the frequency
    density_state = 1 / nq * np.sum(
        1 / math.sqrt(math.pi) / eps * np.exp(-1 * np.power(omg.T - frq, 2) / eps / eps),
        axis=1)[np.newaxis]  # density of state
    return hessian_matrix, frq, density_state, omg


def atoms_position(path_to_atoms_positions, num_atoms, num_atoms_unit_cell, reference_unit_cell, skip_lines=16):
    """
    This function returns the position of atoms, lattice points and positional vectors respect to a reference lattice
    point
    :arg
        path_to_atoms_positions             : Point to the data file in LAMMPS format, string
        num_atoms                           : Number of atoms, integer
        num_atoms_unit_cell                 : Number of atoms per unitcell
        reference_unit_cell                 : The index of the unitcell that is considered as the origin (0, 0, 0),
                                              integer
    :returns
        position_of_atoms                   : Atoms position, [index, atom type, x, y, z], N by 5 array
        lattice_points                      : Lattice point [x, y, z] N/Number of atoms per unitcell by 3 array
        lattice_points_vectors              : Positional vector from origin to the unitcells,
                                              N/Number of atoms per unitcell by 3 array
    """
    with open(os.path.expanduser(path_to_atoms_positions)) as atomsPositionsFile:
        position_tmp_1 = atomsPositionsFile.readlines()
    position_tmp_2 = [line.split() for line in position_tmp_1]
    [position_tmp_2.pop(0) for _ in range(skip_lines)]
    position_tmp_3 = np.array([[float(_) for _ in __] for __ in position_tmp_2[0:num_atoms]])
    position_of_atoms = position_tmp_3[np.argsort(position_tmp_3[:, 0])]
    lattice_points = np.array([_[2:5] for _ in position_of_atoms[::num_atoms_unit_cell]])
    lattice_points_vectors = lattice_points - numpy.matlib.repmat(lattice_points[reference_unit_cell],
                                                                  len(lattice_points), 1)
    return position_of_atoms, lattice_points, lattice_points_vectors


def qpoints(num_qpoints, lattice_parameter, BZ_path):
    """
    This function samples the BZ path
    :arg
        num_qpoints                         : Sampling number, integer
        lattice_parameter                   : Lattice parameter, floating
        BZ_path                                : 1 by 3 list of [1s or 0s], where 1 is yes and 0 is no
    :returns
        points                              : wave vectors
    """
    points = np.multiply(
        numpy.matlib.repmat(np.array([BZ_path]).T, 1, num_qpoints),
        numpy.matlib.repmat(np.linspace(0, math.pi / lattice_parameter, num_qpoints)[np.newaxis], 3, 1)
    )
    return points


def dynamical_matrix(path_to_mass_weighted_hessian, path_to_atoms_positions, num_atoms, num_atoms_unit_cell,
                     central_unit_cell, lattice_parameter, BZ_path=[0, 0, 1], skip_lines=16, num_qpoints=1000):
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
        BZ_path                                : 1 by 3 list of [1s or 0s], where 1 is yes and 0 is no
        num_qpoints                         : Sampling number, integer
    :returns
        eig_vector                          : Eigenvector
        frequency                           : Frequency in rad/S, can be converted to THz using conversion_factor_to_THz
        points                              : BZ path sampling, 3 by num_qpoints array
    """
    hessian_matrix = vibrational_density_state(path_to_mass_weighted_hessian)[0]
    crystal_points = atoms_position(path_to_atoms_positions, num_atoms, num_atoms_unit_cell,
                                    central_unit_cell, skip_lines)
    d_matrix = np.zeros((num_atoms_unit_cell * 3, num_atoms_unit_cell * 3))
    points = qpoints(num_qpoints, lattice_parameter, [0, 0, 1])
    for _ in range(num_qpoints):
        dynamical_matrix_per_qpoint = np.zeros((num_atoms_unit_cell * 3, num_atoms_unit_cell * 3))
        for __ in range(len(crystal_points[2])):
            sum_matrix = hessian_matrix[__ * num_atoms_unit_cell * 3: (__ + 1) * num_atoms_unit_cell * 3,
                         central_unit_cell * num_atoms_unit_cell * 3: (central_unit_cell + 1) *
                                                                      num_atoms_unit_cell * 3] * cmath.exp(
                -1j * np.dot(crystal_points[2][__], points[:, _]))
            dynamical_matrix_per_qpoint = dynamical_matrix_per_qpoint + sum_matrix
        d_matrix = np.append(d_matrix, dynamical_matrix_per_qpoint, axis=0)
    d_matrix = d_matrix[num_atoms_unit_cell * 3:]
    eig_value = np.array([])
    eig_vector = np.zeros((num_atoms_unit_cell * 3, num_atoms_unit_cell * 3))
    for _ in range(num_qpoints):
        dynmat = d_matrix[_ * num_atoms_unit_cell * 3:(_ + 1) * num_atoms_unit_cell * 3]
        eigvals, eigvecs_tmp1, = np.linalg.eigh(dynmat)
        energy = -1 * eigvals.real
        order = np.argsort(energy)
        energy = energy[order]
        eigvecs_tmp2, _ = np.linalg.qr(eigvecs_tmp1)
        eigvecs = np.array([eigvecs_tmp2[_][order] for _ in range(np.shape(eigvecs_tmp2)[1])])
        eig_value = np.append(eig_value, energy).reshape(-1, num_atoms_unit_cell * 3)
        eig_vector = np.append(eig_vector, eigvecs, axis=0)
    eig_vector = eig_vector[num_atoms_unit_cell * 3:]
    frequency = np.sqrt(np.abs(eig_value))
    return eig_vector, frequency, points


def acoustic_phonon(path_to_mass_weighted_hessian, path_to_atoms_positions, num_atoms, num_atoms_unit_cell,
                    central_unit_cell, lattice_parameter, intersection, BZ_path=[0, 0, 1], skip_lines=16,
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
                                                  It is better to be at the center and number of cells should be odd number
                                                  , integer
            lattice_parameter                   : Lattice parameter, floating
            path                                : 1 by 3 list of [1s or 0s], where 1 is yes and 0 is no
            num_qpoints                         : Sampling number, integer
            intersection                        : any possible cross band. This works for one cross band between 3rd and
                                                  fifth bands. It needs modification for other cases (simple).
        :returns
            eigvector                           : Eigenvector
            eigenvalues                         : Frequency in rad/S, can be converted to THz using conversion_factor_to_THz
            frq[-1]                             : BZ path sampling, 3 by num_qpoints array
        """
    frq = dynamical_matrix(path_to_mass_weighted_hessian, path_to_atoms_positions, num_atoms, num_atoms_unit_cell,
                           central_unit_cell, lattice_parameter, BZ_path, skip_lines, num_qpoints)
    transverse_mode_frq = frq[1][:, 0]
    longitudinal_mode_frq = np.concatenate((frq[1][0:intersection, 2], frq[1][intersection:, 4]))
    tmp1 = np.reshape(frq[0][:, 2], (num_qpoints, num_atoms_unit_cell * 3))[0:intersection, :]
    tmp2 = np.reshape(frq[0][:, 4], (num_qpoints, num_atoms_unit_cell * 3))[intersection:, :]
    longitudinal_eigvec = np.concatenate((tmp1.T, tmp2.T), axis=1)
    tmp3 = np.reshape(frq[0][:, 0], (num_qpoints, num_atoms_unit_cell * 3))
    tmp4 = np.reshape(frq[0][:, 1], (num_qpoints, num_atoms_unit_cell * 3))
    angle_x = np.array([np.arctan(np.divide(np.sum(-1 * np.multiply(tmp3, numpy.matlib.repmat(np.array([0, 1, 0]),
                                                                                             int(num_qpoints),
                                                                                             num_atoms_unit_cell)),
                                                  axis=1),
                                           np.sum(-1 * np.multiply(tmp4, numpy.matlib.repmat(np.array([0, 1, 0]),
                                                                                             int(num_qpoints),
                                                                                             num_atoms_unit_cell)),
                                                  axis=1)))])
    angle_y = np.array([np.arctan(np.divide(np.sum(-1 * np.multiply(tmp3, numpy.matlib.repmat(np.array([1, 0, 0]),
                                                                                             int(num_qpoints),
                                                                                             num_atoms_unit_cell)),
                                                  axis=1),
                                           np.sum(-1 * np.multiply(tmp4, numpy.matlib.repmat(np.array([1, 0, 0]),
                                                                                             int(num_qpoints),
                                                                                             num_atoms_unit_cell)),
                                                  axis=1)))])
    transverse_eigvec_x = np.multiply(tmp3, numpy.matlib.repmat(np.cos(angle_x).T, 1, 3 * num_atoms_unit_cell)) + \
                          np.multiply(tmp4, numpy.matlib.repmat(np.sin(angle_x).T, 1, 3 * num_atoms_unit_cell))

    transverse_eigvec_y = np.multiply(tmp3, numpy.matlib.repmat(np.cos(angle_y).T, 1, 3 * num_atoms_unit_cell)) + \
                          np.multiply(tmp4, numpy.matlib.repmat(np.sin(angle_y).T, 1, 3 * num_atoms_unit_cell))

    eigenvalue = np.array([transverse_mode_frq, transverse_mode_frq, longitudinal_mode_frq])
    eigenvector = np.array([transverse_eigvec_x.T, transverse_eigvec_y.T, longitudinal_eigvec])

    return eigenvalue, eigenvector, frq[2]


def gaussian_distribution(amplitude, sigma, wavenumber_idx, num_qpoints, lattice_parameter, BZ_path=[0, 0, 1]):
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
        gaussian                          : Gaussian, 1 by num_qpoints array
    """
    points = qpoints(num_qpoints, lattice_parameter, BZ_path)
    points_norm = LA.norm(points, axis=0)
    expected_value = points_norm[wavenumber_idx]
    gaussian = amplitude * (1.0 / np.sqrt(2 * math.pi) / sigma) * np.exp(
        -1.0 * np.power(((points_norm - expected_value) / np.sqrt(2) / sigma), 2))
    return gaussian[np.newaxis], points_norm[np.newaxis]


def single_wave(path_to_mass_weighted_hessian, path_to_atoms_positions, num_atoms,
                num_atoms_unit_cell, central_unit_cell, amplitude, lattice_parameter, intersection, frq_mode,
                idx_ko, sigma, rep, BZ_path=None, origin_unit_cell=1, skip_lines=[16, 9], num_qpoints=1000):
    """
    This function samples the BZ path
    :arg
        num_qpoints                         : Sampling number, integer
        lattice_parameter                   : Lattice parameter, floating
        BZ_path                                : 1 by 3 list of [1s or 0s], where 1 is yes and 0 is no
    :returns
        points                              : wave vectors
    """
    if BZ_path is None:
        BZ_path = [0, 0, 1]
    num_cell = rep[0] * rep[1] * rep[2]
    positions = atoms_position(path_to_atoms_positions[1], num_atoms[1], num_atoms_unit_cell, origin_unit_cell,
                               skip_lines[1])
    solid_lattice_points = positions[2][:num_cell]
    frq = acoustic_phonon(path_to_mass_weighted_hessian, path_to_atoms_positions[0], num_atoms[0], num_atoms_unit_cell,
                          central_unit_cell, lattice_parameter, intersection, BZ_path, skip_lines[0], num_qpoints)
    gaussian = gaussian_distribution(amplitude, sigma, idx_ko, num_qpoints, lattice_parameter)[0]
    points = qpoints(num_qpoints, lattice_parameter, BZ_path)
    sign_correction = np.exp(
        -1j * (np.arctan(frq[1][frq_mode][frq_mode, idx_ko].imag / frq[1][frq_mode][frq_mode, idx_ko].real)))
    wv_x = gaussian[0, idx_ko] * \
           np.multiply(numpy.matlib.repmat(frq[1][frq_mode][::3, idx_ko][np.newaxis].T, num_cell, 1),
                       np.exp(-1j * np.matmul(np.reshape(np.tile(solid_lattice_points,
                                                                 num_atoms_unit_cell),
                                                         (num_cell * num_atoms_unit_cell, 3)),
                                              points[:, idx_ko][np.newaxis].T))) * sign_correction
    wv_y = gaussian[0, idx_ko] * \
           np.multiply(numpy.matlib.repmat(frq[1][frq_mode][1::3, idx_ko][np.newaxis].T, num_cell, 1),
                       np.exp(-1j * np.matmul(np.reshape(np.tile(solid_lattice_points,
                                                                 num_atoms_unit_cell),
                                                         (num_cell * num_atoms_unit_cell, 3)),
                                              points[:, idx_ko][np.newaxis].T))) * sign_correction
    wv_z = gaussian[0, idx_ko] * \
           np.multiply(numpy.matlib.repmat(frq[1][frq_mode][2::3, idx_ko][np.newaxis].T, num_cell, 1),
                       np.exp(-1j * np.matmul(np.reshape(np.tile(solid_lattice_points,
                                                                 num_atoms_unit_cell),
                                                         (num_cell * num_atoms_unit_cell, 3)),
                                              points[:, idx_ko][np.newaxis].T))) * sign_correction
    wave = np.concatenate((wv_x.T, wv_y.T, wv_z.T), axis=0)
    return positions, frq, gaussian, sign_correction, points, solid_lattice_points, wave.real


def wavepacket(path_to_mass_weighted_hessian, path_to_atoms_positions, num_atoms,
               num_atoms_unit_cell, central_unit_cell, amplitude, lattice_parameter, intersection, frq_mode,
               idx_ko, sigma, rep, time=10, BZ_path=None, origin_unit_cell=0, skip_lines=[16, 9],
               num_qpoints=500):
    """
    This function samples the BZ path
    :arg
        num_qpoints                         : Sampling number, integer
        lattice_parameter                   : Lattice parameter, floating
        BZ_path                                : 1 by 3 list of [1s or 0s], where 1 is yes and 0 is no
    :returns
        points                              : wave vectors
    """
    if BZ_path is None:
        BZ_path = [0, 0, 1]
    num_cell = rep[0] * rep[1] * rep[2]
    positions = atoms_position(path_to_atoms_positions[1], num_atoms[1], num_atoms_unit_cell, origin_unit_cell,
                               skip_lines[1])
    solid_lattice_points = positions[2][:num_cell]
    frq = acoustic_phonon(path_to_mass_weighted_hessian, path_to_atoms_positions[0], num_atoms[0], num_atoms_unit_cell,
                          central_unit_cell, lattice_parameter, intersection, BZ_path, skip_lines[0], num_qpoints)
    points = qpoints(num_qpoints, lattice_parameter, BZ_path)
    gaussian = gaussian_distribution(amplitude, sigma, idx_ko, num_qpoints, lattice_parameter)[0]
    sign_correction = \
    np.exp(-1j * (np.arctan(np.divide(frq[1][frq_mode][frq_mode, :].imag[np.newaxis],
                                      frq[1][frq_mode][frq_mode, :].real[np.newaxis]))))
    omega_time = np.exp(1j * (2 * math.pi) * time * frq[0][frq_mode,:])[np.newaxis]

    wv_x = np.sum(np.multiply(
        np.multiply(
    numpy.matlib.repmat(np.multiply(np.multiply(gaussian, sign_correction),omega_time), num_cell * num_atoms_unit_cell, 1),
    numpy.matlib.repmat(frq[1][frq_mode][::3, :], num_cell, 1)
        ),
        np.exp(-1j * np.matmul(np.reshape(np.tile(solid_lattice_points, num_atoms_unit_cell),
                                      (num_cell * num_atoms_unit_cell, 3)), points)
               )
    ), axis=1)[np.newaxis]

    wv_y = np.sum(np.multiply(
        np.multiply(
    numpy.matlib.repmat(np.multiply(np.multiply(gaussian, sign_correction),omega_time), num_cell * num_atoms_unit_cell, 1),
    numpy.matlib.repmat(frq[1][frq_mode][1::3, :], num_cell, 1)
        ),
        np.exp(-1j * np.matmul(np.reshape(np.tile(solid_lattice_points, num_atoms_unit_cell),
                                      (num_cell * num_atoms_unit_cell, 3)), points)
               )
    ), axis=1)[np.newaxis]

    wv_z = np.sum(np.multiply(
        np.multiply(
    numpy.matlib.repmat(np.multiply(np.multiply(gaussian, sign_correction),omega_time), num_cell * num_atoms_unit_cell, 1),
    numpy.matlib.repmat(frq[1][frq_mode][2::3, :], num_cell, 1)
        ),
        np.exp(-1j * np.matmul(np.reshape(np.tile(solid_lattice_points, num_atoms_unit_cell),
                                      (num_cell * num_atoms_unit_cell, 3)), points)
               )
    ), axis=1)[np.newaxis]
    phonon_wavepacket = np.concatenate((wv_x, wv_y, wv_z), axis=0)
    # return num_cell, positions, solid_lattice_points, frq, points, gaussian, sign_correction, omega_time
    return phonon_wavepacket, positions

class ITC:
    """
    ITC is a class written in python to calculate interfacial thermal conductance of two solids in contact.

    """

    def __init__(self, rho, c):
        """
        :arg
            rho             : mass density
            c               : speed of sound

        :type
            rho             : 1 by 2 np.array
            c               : 1 by 2 np.array
        """
        self.rho = rho
        self.c = c

    def acoustic_mismatch(self, n_mu=2e4, n_sf=5e4):
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
            mu_crt          : critical angle, float number
            tij             : angle independent transmission coefficient from i to j, np.array
            Tij             : transmission coefficient from i to j, np.array
            MTij            : matrix of transmission coefficient, row: freq dependency
                                                    column: angle dependency, np.array
        """
        mu_crt = math.cos(math.asin(self.c[1] / self.c[0]))
        z_i = self.rho[0] * self.c[0]
        z_j = self.rho[1] * self.c[1]
        mu_i = np.linspace(0, 1, n_mu)
        mu_j = np.sqrt(1 - ((self.c[1] / self.c[0]) ** 2) * (1 - np.power(mu_i, 2)))
        tij = 4 * (z_i * z_j) * np.divide(np.multiply(mu_i, mu_j), np.power(z_i * mu_i + z_j * mu_j, 2))
        tij[mu_i < mu_crt] = 0
        # tij = np.trapz(y=Tij, x=mu_i, dx=mu_i[1] - mu_i[0], axis=-1)
        # omg = np.linspace(0, omega_max, n_omg)
        # cutoff_idx = np.where(omg > omega_cutoff)
        # MTij = np.tile(Tij, (np.shape(omg)[0], 1))
        # MTij[cutoff_idx[0][0]:] = 0
        sf_x = np.linspace(mu_crt, 1, n_sf)
        tmp_1 = np.sqrt(1 - (self.c[1] / self.c[0]) ** 2 * (1 - np.power(sf_x, 2)))
        tmp_2 = np.power((z_i / z_j + np.divide(tmp_1, sf_x)), 2)
        sf_y = (1 + z_i / z_j) ** 2 * np.divide(tmp_1, tmp_2)
        suppression_function = np.trapz(sf_y, sf_x)
        return mu_i, mu_j, mu_crt, tij, suppression_function

    def diffuse_mismatch(self, path_to_mass_weighted_hessian, eps, nq, nsampleing):
        tmp_1 = vibrational_density_state(path_to_mass_weighted_hessian[0], eps[0], nq[0])
        tmp_2 = vibrational_density_state(path_to_mass_weighted_hessian[1], eps[1], nq[1])
        omg_max = np.min([np.max(tmp_1[-1]), np.max(tmp_2[-1])])
        omg = np.linspace(0, omg_max, nsampleing)
        f_i = interpolate.PchipInterpolator(tmp_1[-1], tmp_1[2], extrapolate=None)
        f_j = interpolate.PchipInterpolator(tmp_2[-1], tmp_2[2], extrapolate=None)
        dos_i = f_i(omg)
        dos_j = f_j(omg)
        tij = np.divide(self.c[1] * dos_j, self.c[0] * dos_i + self.c[1] * dos_j, out=np.zeros_like(self.c[1] * dos_j),
                        where=self.c[0] * dos_i + self.c[1] * dos_j != 0)
        return tij, omg

    # def equilibrium_thermal_conductance(self):


# vDoS = vibrational_density_state("~/Desktop/ITC/Run-00-si-heavy-si/Run-01-hessian/Si-hessian-mass-weighted-hessian.d",
#                                  eps=3e12, nq=1e3)
#
# plt.figure()
# plt.plot(vDoS[3][0], vDoS[2][0])
# plt.show()
#
# plt.figure()
# ax = sns.heatmap(vDoS[0].real)
#
# plt.figure()
# ax = sns.heatmap(vDoS[0].imag)
#
# plt.figure()
# plt.plot(vDoS[1][0])
# plt.show()
#
# plt.figure()
# plt.plot(vDoS[3][0])
# plt.show()

# position_wrapped = atoms_position("~/Desktop/ITC/Run-00-si-heavy-si/Run-00-configuration-add-heavy-si/data.SiSi",
#                                   160000, 8, 0, skip_lines=16)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(position_wrapped[0][:, 2], position_wrapped[0][:, 3], position_wrapped[0][:, 4])
#
# position_unwrapped = atoms_position("~/Desktop/ITC/Run-00-si-heavy-si/Run-00-configuration-add-heavy-si/data.unwraped",
#                                     160000, 8, 0, skip_lines=9)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(position_unwrapped[0][:, 2], position_unwrapped[0][:, 3], position_unwrapped[0][:, 4])
#
# points = qpoints(500, 5.43, [0, 0, 1])
# plt.plot(points[0])
# plt.plot(points[1])
# plt.plot(points[2])
#
# dynam_matrix = dynamical_matrix("~/Desktop/ITC/Run-00-si-heavy-si/Run-01-hessian/Si-hessian-mass-weighted-hessian.d",
#                                 "~/Desktop/ITC/Run-00-si-heavy-si/Run-01-hessian/data.Si-unwraped",
#                                 1000, 8, 63, 5.43, skip_lines=9)
# print(np.shape(dynam_matrix[0]))
#
# fig = plt.figure()
# ax = sns.heatmap(np.reshape(dynam_matrix[0][:, 2], (1000, 8 * 3)).real.T)
#
# fig = plt.figure()
# plt.plot(dynam_matrix[1])
# plt.show()
# fig = plt.figure()
# plt.plot(dynam_matrix[1][:, 0], 'b--', dynam_matrix[1][:, -1], 'r--')
# plt.show()
# matrix = acoustic_phonon("~/Desktop/ITC/Run-00-si-heavy-si/Run-01-hessian/Si-hessian-mass-weighted-hessian.d",
#                          "~/Desktop/ITC/Run-00-si-heavy-si/Run-01-hessian/data.Si-unwraped",
#                          1000, 8, 63, 5.43, 389, num_qpoints=500, skip_lines=9)
# print(np.shape(matrix[2]))
# plt.plot(LA.norm(matrix[2], axis=0), matrix[0][0], '--', LA.norm(matrix[2], axis=0), matrix[0][1])
# plt.show()
#
# plt.figure()
# ax = sns.heatmap(matrix[1][0].real)
#
# plt.figure()
# ax = sns.heatmap(matrix[1][1].real)
#
# plt.figure()
# ax = sns.heatmap(matrix[1][2].real)
#
# plt.figure()
# ax = sns.heatmap(matrix[1][0].imag.T)
#
# Gaussian = gaussian_distribution(0.0005, 0.005, 300, 1000, 5.43)
# plt.plot(Gaussian[1][0], Gaussian[0][0])

# wave = single_wave("~/Desktop/ITC/Run-00-si-heavy-si/Run-01-hessian/Si-hessian-mass-weighted-hessian.d",
#                    ["~/Desktop/ITC/Run-00-si-heavy-si/Run-01-hessian/data.Si-5x5x5",
#                     "~/Desktop/ITC/Run-00-si-heavy-si/Run-00-configuration-add-heavy-si/data.unwraped"],
#                    [1000, 8 * 5 * 5 * 400 + 8 * 5 * 5 * 400], 8, 63, 0.0005, 5.43, 390, 2, 50, 0.005, [5, 5, 400],
#                    origin_unit_cell=1, num_qpoints=501)
# plt.plot(wave[-2][:,2],wave[-1][2,::8])
wv = wavepacket("~/Desktop/ITC/Run-00-si-heavy-si/Run-01-hessian/Si-hessian-mass-weighted-hessian.d",
                ["~/Desktop/ITC/Run-00-si-heavy-si/Run-01-hessian/data.Si-5x5x5",
                 "~/Desktop/ITC/Run-00-si-heavy-si/Run-00-configuration-add-heavy-si/data.unwraped"],
                [1000, 8 * 5 * 5 * 400 + 8 * 5 * 5 * 400], 8, 63, 0.005, 5.43, 391, 1, 140, 0.05, [5, 5, 400],
                time=0)
plt.plot(wv[1][0][:80000,4],wv[0][2,:].real)
# A = ITC(rho=[1.2, 1.2], c=[2, 1])
# B = A.acoustic_mismatch()
# plt.polar(np.arccos(B[0]), B[-2], 'o')
# plt.show()

# C = A.diffuse_mismatch(["~/Desktop/cleanUpDesktop/LamResearch_Internship_Summer_2019/Run-14-hessian-analysis/"
#                         "Run-01-Si-28"
#                         "-Si-72/Si-hessian-mass-weighted-hessian.d",
#                         "~/Desktop/cleanUpDesktop/LamResearch_Internship_Summer_2019/"
#                         "Run-14-hessian-analysis/Run-06-Ge/Si-hessian-mass-weighted-hessian.d"],
#                        eps=[3e12, 3e12], nq=[1e4, 1e4], nsampleing=1e4)
