"""
Developed by S. Aria Hosseini
Department of mechanical engineering
University of California, Riverside
"""
import numpy as np


def acoustic_mismatch(speed_of_sound,  mass_density, n_mu=2e4, n_sf=5e4):

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

        mu_crt = np.cos(math.asin(self.c[1] / self.c[0]))
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
