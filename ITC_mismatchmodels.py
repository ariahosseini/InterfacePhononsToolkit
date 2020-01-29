"""
Developed by S. Aria Hosseini
Department of mechanical engineering
University of California, Riverside
"""
import numpy as np
import numpy.matlib
from scipy import interpolate
import math
import cmath
import os
from sys import exit
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.core.multiarray import ndarray

global conversion_factor_to_THz
conversion_factor_to_THz = 15.633302


def vibrational_density_state(path_to_mass_weighted_hessian, eps=3e12, nq=2e4):
    """
    This function calculate vibrational density of state from hessian matrix
    of molecular dynamics calculations
    :arg
        path_to_mass_weighted_hessian       : point to the mass weighted hessian file
        eps                                 : tuning parameter, relate to line width
        nq                                  : sampling grid

    :type
        path_to_mass_weighted_hessian       : string
        eps                                 : float number
        nq                                  : integer number

    :returns
        hessian_matrix                      : hessian matrix, np.array
        frq                                 : frequency in 1/cm
        density_state                       : vibrational density of state
        omg                                 : frequency sampling corresponds to "density_state"
    """
    with open(os.path.expanduser(path_to_mass_weighted_hessian)) as hessian_file:
        hm_tmp_1 = hessian_file.readlines()  # hm stands for hessian matrix
    hm_tmp_2 = [line.split() for line in hm_tmp_1]
    hm_tmp_3 = np.array([[float(_) for _ in __] for __ in hm_tmp_2])
    hessian_file.close()
    hessian_symmetry = (np.triu(hm_tmp_3) + np.tril(hm_tmp_3).transpose()) / 2  # Hessian matrix is Hermitian
    hessian_matrix = hessian_symmetry + np.triu(hessian_symmetry, 1).transpose()
    egn_value, egn_vector = np.linalg.eigh(hessian_matrix)
    egn_value = np.where(egn_value < 0, egn_value, 0)
    frq = np.sqrt(-1 * egn_value)  # Frequency from Hessian matrix
    frq = np.sort(frq)
    omg = np.linspace(np.min(frq), np.max(frq), nq)  # Sampling the frequency
    density_state = np.sum(1 / math.sqrt(math.pi) / eps * np.exp(-1 * np.power(np.array([omg]).T - frq, 2) / eps / eps),
                           axis=1)  # density of state
    return hessian_matrix, frq, density_state, omg


def atoms_position(path_to_atoms_positions, num_atoms, num_atoms_unit_cell, central_unit_cell, skip_lines=16):
    with open(os.path.expanduser(path_to_atoms_positions)) as atomsPositionsFile:
        position_tmp_1 = atomsPositionsFile.readlines()
    position_tmp_2 = [line.split() for line in position_tmp_1]
    [position_tmp_2.pop(0) for _ in range(skip_lines)]
    position_tmp_3 = np.array([[float(_) for _ in __] for __ in position_tmp_2[0:num_atoms]])
    position_of_atoms = position_tmp_3[np.argsort(position_tmp_3[:, 0])]
    lattice_points = np.array([_[2:5] for _ in position_of_atoms[::num_atoms_unit_cell]])
    lattice_points_vectors = lattice_points - numpy.matlib.repmat(lattice_points[central_unit_cell],
                                                                  len(lattice_points), 1)
    return position_of_atoms, lattice_points, lattice_points_vectors


def qpoints(num_qpoints, lattice_parameter):
    points = np.array([np.zeros(num_qpoints), np.zeros(num_qpoints),
                       np.linspace(-math.pi / lattice_parameter, math.pi / lattice_parameter, num=num_qpoints)])
    return points


def dynamical_matrix(path_to_mass_weighted_hessian, path_to_atoms_positions, num_atoms, num_atoms_unit_cell,
                     central_unit_cell, lattice_parameter, skip_lines=16, num_qpoints=1000):
    hessian_matrix = vibrational_density_state(path_to_mass_weighted_hessian)[0]
    crystal_points = atoms_position(path_to_atoms_positions, num_atoms, num_atoms_unit_cell,
                                    central_unit_cell, skip_lines)
    d_matrix = np.zeros((num_atoms_unit_cell * 3, num_atoms_unit_cell * 3))
    points = qpoints(num_qpoints, lattice_parameter)
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
    return eig_vector, frequency


def acoustic_phonon(path_to_mass_weighted_hessian, path_to_atoms_positions, num_atoms, num_atoms_unit_cell,
                    central_unit_cell, lattice_parameter, intersection, skip_lines=16, num_qpoints=1000):
    frq = dynamical_matrix(path_to_mass_weighted_hessian, path_to_atoms_positions, num_atoms, num_atoms_unit_cell,
                           central_unit_cell, lattice_parameter, skip_lines=16, num_qpoints=1000)
    transverse_mode_frq = frq[1][int(num_qpoints//2):, 0]
    longitudinal_mode_frq = np.concatenate((frq[1][int(num_qpoints//2):intersection, 2], frq[1][intersection:, 4]))
    tmp1 = np.reshape(frq[0][:, 2], (num_qpoints, num_atoms_unit_cell * 3))[int(num_qpoints//2):intersection, :]
    tmp2 = np.reshape(frq[0][:, 4], (num_qpoints, num_atoms_unit_cell * 3))[intersection:, :]
    longitudinal_eigvec = np.concatenate((tmp1.T, tmp2.T), axis=1)
    return transverse_mode_frq, longitudinal_mode_frq, longitudinal_eigvec


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


A = ITC(rho=[1.2, 1.2], c=[2, 1])
# B = A.acoustic_mismatch()
# plt.polar(np.arccos(B[0]), B[-2], 'o')
# plt.show()

# print(B[-1])

# exit(0)                              # Successful exit
#
#
# vDoS = vibrational_density_state("~/Desktop/cleanUpDesktop/LamResearch_Internship_Summer_2019/"
#                                  "Run-14-hessian-analysis/"
#                                  "Run-01-Si-28"
#                                  "-Si-72/Si-hessian-mass-weighted-hessian.d")

# plt.plot(vDoS[3], vDoS[2])
# plt.show()

# ax = sns.heatmap(vDoS[0].real, linewidth=0.5)
# plt.show()
#
# ax = sns.heatmap(vDoS[0].imag, linewidth=0.5)
# plt.show()

# plt.plot(vDoS[1])
# plt.show()

# B = atoms_position('~/Desktop/cleanUpDesktop/LamResearch_Internship_Summer_2019/'
#                    'Run-14-hessian-analysis/Run-05-Si/data.Si-5x5x5', 1000, 8, 63, skip_lines=16)
# print(B[2])
# C = A.diffuse_mismatch(["~/Desktop/cleanUpDesktop/LamResearch_Internship_Summer_2019/Run-14-hessian-analysis/"
#                         "Run-01-Si-28"
#                         "-Si-72/Si-hessian-mass-weighted-hessian.d",
#                         "~/Desktop/cleanUpDesktop/LamResearch_Internship_Summer_2019/"
#                         "Run-14-hessian-analysis/Run-06-Ge/Si-hessian-mass-weighted-hessian.d"],
#                        eps=[3e12, 3e12], nq=[1e4, 1e4], nsampleing=1e4)
# plt.plot(C[1], C[0])
# plt.show()

# matrix = dynamical_matrix("~/Desktop/cleanUpDesktop/LamResearch_Internship_Summer_2019/"
#                           "Run-14-hessian-analysis/Run-06-Ge/Si-hessian-mass-weighted-hessian.d",
#                           '~/Desktop/cleanUpDesktop/LamResearch_Internship_Summer_2019/'
#                           'Run-14-hessian-analysis/Run-05-Si/data.Si-5x5x5', 1000, 8, 63, 5.43, skip_lines=16)
# print(np.shape(matrix[0]))
# np.shape(matrix[0])
# ax = sns.heatmap(np.reshape(matrix[0][12000:, 2].real, (500, 24)).T)

# plt.plot(matrix[1][:, -1], 'r--', matrix[1][:, 0])
# plt.show()
#
matrix = acoustic_phonon("~/Desktop/cleanUpDesktop/LamResearch_Internship_Summer_2019/"
                         "Run-14-hessian-analysis/Run-06-Ge/Si-hessian-mass-weighted-hessian.d",
                         '~/Desktop/cleanUpDesktop/LamResearch_Internship_Summer_2019/'
                         'Run-14-hessian-analysis/Run-05-Si/data.Si-5x5x5', 1000, 8, 63, 5.43, 900, skip_lines=16)
# print(np.shape(matrix[2]))
# plt.plot(matrix[0], '--', matrix[1])
# plt.show()

ax = sns.heatmap(matrix[2].real)
# print(np.shape(matrix[0][:24, :]), np.shape(matrix), np.shape(np.reshape(matrix[0][:, 0].real, (1000, 24))))
#
# a = np.reshape(matrix[0][:, 0].real, (1000, 24))
# print(a, a[0], a[:,0], np.shape(a), a[:][0])

# ax = sns.heatmap(AZ2[0].real, linewidth=0.5)
# plt.show()
# plt.polar(A[0], A[1])
# plt.show()
# plt.plot(A[0], A[1])
# plt.show()
