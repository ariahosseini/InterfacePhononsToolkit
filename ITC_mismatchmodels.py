"""
Developed by S. Aria Hosseini
Department of mechanical engineering
University of California, Riverside
"""
import numpy as np
import numpy.matlib
import math
import cmath
import os
import matplotlib.pyplot as plt
import seaborn as sns


def vibrational_density_state(path_to_mass_weighted_hessian, eps=3e12, nq=1e4):
    """
    This function calculate vibrational density of state from hessian matrix
    of molecular dynamics calculations
    :arg
        path_to_mass_weighted_hessian    : point to the mass weighted hessian file
        eps                              : tuning parameter, relate to line width
        nq                               : sampling grid

    :type
       path_to_mass_weighted_hessian        : string
       eps                                  : float number
       nq                                   : integer number

    :returns
        hessian_matrix                      : hessian matrix, np.array
        frq,                                : frequency in 1/cm
        density_state                       : vibrational density of state
    """
    with open(os.path.expanduser(path_to_mass_weighted_hessian)) as hessian_file:
        hm_tmp_1 = hessian_file.readlines()
    hm_tmp_2 = [line.split() for line in hm_tmp_1]
    hm_tmp_3 = np.array([[float(_) for _ in __] for __ in hm_tmp_2])
    hessian_file.close()
    hessian_symmetry = (np.triu(hm_tmp_3) + np.tril(hm_tmp_3).transpose()) / 2
    hessian_matrix = hessian_symmetry + np.triu(hessian_symmetry, 1).transpose()
    egn_value, egn_vector = np.linalg.eigh(hessian_matrix)
    egn_value = np.where(egn_value < 0, egn_value, 0)
    frq = np.sqrt(-1 * egn_value)
    frq = np.sort(frq)
    omg = np.linspace(np.min(frq), np.max(frq), nq)
    density_state = np.sum(1 / math.sqrt(math.pi) / eps * np.exp(-1 * np.power(np.array([omg]).T - frq, 2) / eps / eps),
                           axis=1)
    return hessian_matrix, frq, density_state


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
        dynam_matrix_per_qpoint = np.zeros((num_atoms_unit_cell * 3, num_atoms_unit_cell * 3))
        for __ in range(len(crystal_points[2])):
            sum_matrix = hessian_matrix[__ * num_atoms_unit_cell * 3: (__ + 1) * num_atoms_unit_cell * 3,
                                        central_unit_cell * num_atoms_unit_cell * 3: (central_unit_cell + 1) *
                                        num_atoms_unit_cell * 3] * cmath.exp(
                                                             -1j * np.dot(crystal_points[2][__], points[:, _]))
            dynam_matrix_per_qpoint = dynam_matrix_per_qpoint + sum_matrix
        d_matrix = np.append(d_matrix, dynam_matrix_per_qpoint, axis=0)
    d_matrix = d_matrix[num_atoms_unit_cell * 3:]
    eig_value = np.array([])
    eig_vector = np.zeros((num_atoms_unit_cell * 3, num_atoms_unit_cell * 3))
    for _ in range(num_qpoints):
        dynmat = d_matrix[_ * num_atoms_unit_cell * 3:(_ + 1) * num_atoms_unit_cell * 3]
        eigvals, eigvecs, = np.linalg.eigh(dynmat)
        eig_value = np.append(eig_value, eigvals).reshape(-1, num_atoms_unit_cell * 3)
        eig_vector = np.append(eig_vector, eigvecs, axis=0)
    eig_vector = eig_vector[num_atoms_unit_cell * 3:]
    frequency = np.sqrt(np.abs(-1*eig_value.real))
    conversion_factor_to_THz = 15.633302
    frequency = frequency * conversion_factor_to_THz
    return eig_vector, frequency


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

    def acoustic_mismatch(self, omega_cutoff, omega_max, n_omg=1000, n_mu=2000):
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
        Tij = 4 * (z_i * z_j) * np.divide(np.multiply(mu_i, mu_j), np.power(z_i * mu_i + z_j * mu_j, 2))
        Tij[mu_i < mu_crt] = 0
        tij = np.trapz(y=Tij, x=mu_i, dx=mu_i[1] - mu_i[0], axis=-1)
        omg = np.linspace(0, omega_max, n_omg)
        cutoff_idx = np.where(omg > omega_cutoff)
        MTij = np.tile(Tij, (np.shape(omg)[0], 1))
        MTij[cutoff_idx[0][0]:] = 0
        return mu_i, mu_j, mu_crt, tij, Tij, MTij

    def diffuse_mismatch(self, path_to_mass_weighted_hessian, eps, nq):
        cdos_i = self.c[0] * vibrational_density_state(path_to_mass_weighted_hessian[0], eps[0], nq[0])[2]
        print(cdos_i, type(cdos_i), np.shape(cdos_i))
        cdos_j = self.c[1] * vibrational_density_state(path_to_mass_weighted_hessian[1], eps[1], nq[1])[2]
        tij = np.array([cdos_j / (cdos_i + cdos_j)])
        return tij


A = ITC(rho=[1.2, 1.2], c=[2, 1])
# B = atoms_position('~/Desktop/cleanUpDesktop/LamResearch_Internship_Summer_2019/'
#                    'Run-14-hessian-analysis/Run-05-Si/data.Si-5x5x5', 1000, 8, 63, skip_lines=16)
# print(B[2])
# C = A.diffuse_mismatch(["~/Desktop/cleanUpDesktop/LamResearch_Internship_Summer_2019/Run-14-hessian-analysis/"
#                         "Run-01-Si-28"
#                         "-Si-72/Si-hessian-mass-weighted-hessian.d",
#                         "~/Desktop/cleanUpDesktop/LamResearch_Internship_Summer_2019/"
#                         "Run-14-hessian-analysis/Run-06-Ge/Si-hessian-mass-weighted-hessian.d"],
#                        eps=[3e12, 3e12], nq=[1e4, 1e4])
matrix = dynamical_matrix("~/Desktop/cleanUpDesktop/LamResearch_Internship_Summer_2019/"
                          "Run-14-hessian-analysis/Run-06-Ge/Si-hessian-mass-weighted-hessian.d",
                          '~/Desktop/cleanUpDesktop/LamResearch_Internship_Summer_2019/'
                          'Run-14-hessian-analysis/Run-05-Si/data.Si-5x5x5', 1000, 8, 63, 5.43, skip_lines=16)
AW = np.expand_dims(matrix[0], axis=0)
AZ = np.reshape(AW, (24, 1000, 24))
AZ2 = np.reshape(AW, (1000, 24, 24))
# print(np.shape(matrix[1]))
# plt.plot(matrix[1])
# plt.show()

print(np.shape(matrix[0][:24, :]), np.shape(matrix), np.shape(np.reshape(matrix[0][:, 0].real, (1000, 24))))

a = np.reshape(matrix[0][:, 0].real, (1000, 24))
# print(a, a[0], a[:,0], np.shape(a), a[:][0])

# ax = sns.heatmap(AZ[0].real, linewidth=0.5)
sns.heatmap(a.T)
plt.show()

# ax = sns.heatmap(AZ2[0].real, linewidth=0.5)
# plt.show()
# plt.polar(A[0], A[1])
# plt.show()
# plt.plot(A[0], A[1])
# plt.show()
