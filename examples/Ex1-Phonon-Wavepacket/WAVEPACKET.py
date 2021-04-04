# Import wavepacket_simulator lib
from util.wavepacket_simulator import *

# Compute hessian matrix, frequency, phonon density of state for given frequency range,
# frequency range for the density of state
hessian_matrix, frq, density_state, omg = vibrational_density_state(path_to_mass_weighted_hessian =
                                                                    'Datafiles/Si-hessian-mass-weighted-hessian.d',
                                                                    eps = 3e12, nq = 2e4)

# Read position of atoms from LAMMPS data file
position_of_atoms, lattice_points, lattice_points_vectors = atoms_position(path_to_atoms_positions = 'Datafiles/data.Si-unwrapped',
                                                                           num_atoms = 3 * 3 * 3 * 2, num_atoms_unit_cell = 2,
                                                                           reference_unit_cell = int(3 * 3 * 3 / 2), skip_lines = 16)

# Compute dynamical matrix, eigenvectors phonon dispersion and lattice vectors
eig_vector, eig_value, frequency, points = dynamical_matrix(path_to_mass_weighted_hessian='Datafiles/Si-hessian-mass-weighted-hessian.d',
                                                            path_to_atoms_positions='Datafiles/data.Si-unwrapped', num_atoms=3 * 3 * 3 * 2,
                                                            num_atoms_unit_cell=2, central_unit_cell=int(3 * 3 * 3 / 2),
                                                            lattice_parameter=5.4320, BZ_path=[0, 0, 1], skip_lines=9,
                                                            num_qpoints=1000)

# Compute dynamical matrix, eigenvectors phonon dispersion and lattice vectors for acoustic modes
# Optical modes are neglected
# Degeneracy is handled by projecting eigenvectors along Cartesian x, y directions
eigenvalue, eigenvector, kpoints = acoustic_phonon(path_to_mass_weighted_hessian='Datafiles/Si-hessian-mass-weighted-hessian.d',
                                                   path_to_atoms_positions='Datafiles/data.Si-unwrapped', num_atoms=3 * 3 * 3 * 2,
                                                   num_atoms_unit_cell=2, central_unit_cell=int(3 * 3 * 3 / 2),
                                                   lattice_parameter=5.4320, BZ_path=[0, 0, 1], skip_lines=9,
                                                   num_qpoints=1000)
# Generate a Gaussian distribution
Gaussian, points = gaussian_distribution(amplitude=0.0005, sigma=0.005, wavenumber_idx=300, num_qpoints=1000,
                                         lattice_parameter=5.4320, BZ_path=[0, 0, 1])

# Model a single wave
positions, frq, gaussian, points, solid_lattice_points, wave = single_wave(path_to_mass_weighted_hessian='Datafiles/Si-hessian-mass-weighted-hessian.d',
                                                                           path_to_atoms_positions=['Datafiles/data.Si-unwrapped', 'Datafiles/data.unwrapped'],
                                                                           num_atoms=[54, 160000], num_atoms_unit_cell=2, central_unit_cell=int(3 * 3 * 3 / 2),
                                                                           lattice_parameter=5.4320, frq_mode=2, idx_ko=80, amplitude=0.0005, sigma=0.005,
                                                                           replication=[10, 10, 800], BZ_path=None, origin_unit_cell=0, skip_lines=[9, 9],
                                                                           num_qpoints=1000)

# Model a wavepacket
phonon_wavepacket, positions = wavepacket(path_to_mass_weighted_hessian='Datafiles/Si-hessian-mass-weighted-hessian.d',
                                          path_to_atoms_positions=['Datafiles/data.Si-unwrapped', 'Datafiles/data.unwrapped'],
                                          num_atoms=[54, 160000], num_atoms_unit_cell=2, central_unit_cell=int(3 * 3 * 3 / 2),
                                          lattice_parameter=5.4320, frq_mode=2, idx_ko=80, amplitude=0.0005, sigma=0.005,
                                          replication=[10, 10, 800], BZ_path=None, origin_unit_cell=0, skip_lines=[9, 9],
                                          num_qpoints=1000)
