from utils.func import *
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
# hessian_matrix, frq, density_state, omg = vibrational_density_state(path_to_mass_weighted_hessian='Si-hessian-mass-weighted-hessian.d', eps=3e12, nq=2e4)
# print(np.shape(hessian_matrix), np.shape(frq), np.shape(density_state), np.shape(omg))

# position_of_atoms, lattice_points, lattice_points_vectors = atoms_position(path_to_atoms_positions='data.Si-unwrapped', num_atoms=3 * 3 * 3 * 2, num_atoms_unit_cell=2, reference_unit_cell=int(3 * 3 * 3 / 2), skip_lines=16)
# print(np.shape(position_of_atoms), np.shape(lattice_points), np.shape(lattice_points_vectors))
# print(position_of_atoms)
# exit()
# points = qpoints(num_qpoints=50, lattice_parameter=5.43, BZ_path=[0, 0, 1])
# print(points)

# eig_vector, frequency, points = dynamical_matrix(path_to_mass_weighted_hessian='Si-hessian-mass-weighted-hessian.d', path_to_atoms_positions='data.Si-unwrapped', num_atoms=3 * 3 * 3 * 2, num_atoms_unit_cell=2, central_unit_cell=int(3 * 3 * 3 / 2), lattice_parameter=5.4320, BZ_path=[0, 0, 1], skip_lines=9, num_qpoints=1000)

# print(np.shape(eig_vector), np.shape(frequency), np.shape(points))

# eigenvalue, eigenvector, kpoints = acoustic_phonon(path_to_mass_weighted_hessian='Si-hessian-mass-weighted-hessian.d', path_to_atoms_positions='data.Si-unwrapped', num_atoms=3 * 3 * 3 * 2, num_atoms_unit_cell=2, central_unit_cell=int(3 * 3 * 3 / 2), lattice_parameter=5.4320, BZ_path=[0, 0, 1], skip_lines=9, num_qpoints=1000)

# print(np.shape(eigenvalue), np.shape(eigenvector), np.shape(kpoints))
# plt.plot(kpoints[-1], eigenvalue[0])
# plt.plot(kpoints[-1], eigenvalue[1])
# plt.plot(kpoints[-1], eigenvalue[2])
# plt.show()

# Gaussian, points = gaussian_distribution(amplitude=0.0005, sigma=0.005, wavenumber_idx=300, num_qpoints=1000, lattice_parameter=5.4320, BZ_path=[0, 0, 1])
# print(np.shape(Gaussian), np.shape(points))
# plt.plot(points[0], Gaussian[0])
# plt.show()


# positions, frq, gaussian, points, solid_lattice_points, wave = single_wave(path_to_mass_weighted_hessian='Si-hessian-mass-weighted-hessian.d', path_to_atoms_positions=['data.Si-unwrapped', 'data.unwrapped'], num_atoms=[54, 160000], num_atoms_unit_cell=2, central_unit_cell=int(3 * 3 * 3 / 2), lattice_parameter=5.4320, frq_mode=2, idx_ko=80, amplitude=0.0005, sigma=0.005, replication=[10, 10, 800], BZ_path=None, origin_unit_cell=0, skip_lines=[9, 9], num_qpoints=1000)

# print(np.shape(positions), np.shape(frq), np.shape(gaussian), np.shape(points), np.shape(solid_lattice_points), np.shape(wave))
# position_sorted = positions[positions[:, -1].argsort()]
# print(np.shape(position_sorted))

# wave_sorted = np.concatenate(
#     (wave[0, positions[:, -1].argsort()][np.newaxis],
#      wave[1, positions[:, -1].argsort()][np.newaxis],
#      wave[2, positions[:, -1].argsort()][np.newaxis]),
#     axis=0)

# print(np.shape(wave))
# print(np.shape(wave_sorted))

# plt.figure()
# plt.plot(position_sorted[:, -1], wave_sorted[2, :])
# plt.show()


phonon_wavepacket, positions = wavepacket(path_to_mass_weighted_hessian='Si-hessian-mass-weighted-hessian.d', path_to_atoms_positions=['data.Si-unwrapped', 'data.unwrapped'], num_atoms=[54, 160000], num_atoms_unit_cell=2, central_unit_cell=int(3 * 3 * 3 / 2), lattice_parameter=5.4320, frq_mode=2, idx_ko=80, amplitude=0.0005, sigma=0.005, replication=[10, 10, 800], BZ_path=None, origin_unit_cell=0, skip_lines=[9, 9], num_qpoints=1000)
print(np.shape(phonon_wavepacket), np.shape(positions))

plt.figure()
plt.plot(positions[:, -1], phonon_wavepacket[0, :])
plt.show()

plt.figure()
plt.plot(positions[:, -1], phonon_wavepacket[1, :])
plt.show()

plt.figure()
plt.plot(positions[:, -1], phonon_wavepacket[2, :])
plt.show()
