import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig

def wavepacket_simulator_lg(n_kp, lc, n_cel, M, pos, idx_ko_matrix, amp_matrix, nshift):
    """
    Function to calculate phonon wave packet and frequency bands for longitudinal mode.
    
    Parameters:
    n_kp           : int    : Number of K-points
    lc             : float  : Latttice Cte.
    n_cel          : int    : Number of cells
    M              : ndarray: Noisy Hessian matrix
    pos            : ndarray: Atom positions sorted by id
    idx_ko_matrix  : list   : Indices for wave vector centers
    amp_matrix     : list   : Amplitudes for Gaussian wave packet
    nshift         : int    : Shift for longitudinal acoustic mode
    
    Returns:
    freq           : ndarray: Frequencies in THz
    egn_vec_l      : ndarray: Eigenvectors for longitudinal mode
    """

    # define kpoints
    kp = np.vstack([np.zeros((2, n_kp)), np.linspace(0, np.pi / lc, n_kp)])  # wavevector
    kp_norm = np.linalg.norm(kp, axis=0)

    # ensure Hessian is symmetric
    H = (np.triu(M) + np.tril(M).T) / 2  
    Hsn = np.triu(H) + np.triu(H, 1).T

    # sort atom positions by id
    po = pos[np.argsort(pos[:, 0])]  
    p = np.zeros((pos.shape[0], 8))
    p[:, :5] = pos
    n_atm_uitcel = 8

    lp = p[::n_atm_uitcel, 2:5]  # lattice points
    ctr_cell = (n_cel ** 3 // 2) + 1  # central unit cell
    r = lp - lp[ctr_cell - 1]  # relative positions to central cell
    dynam_sz = 3 * n_atm_uitcel  # size of dynamical matrix
    dynam = np.zeros((n_kp, dynam_sz, dynam_sz), dtype=complex)

    # build the dynamical matrix
    for i in range(n_kp):
        inloop_var = np.zeros((dynam_sz, dynam_sz), dtype=complex)
        for ii in range(len(r)):
            inlpvar = Hsn[dynam_sz * ii:dynam_sz * (ii + 1),
                          dynam_sz * (ctr_cell - 1):dynam_sz * ctr_cell] \
                * np.exp(-1j * np.dot(r[ii], kp[:, i]))
            inloop_var += inlpvar
        dynam[i] = inloop_var

    # calculate frequencies and eigenvectors
    freq = np.zeros((n_kp, dynam_sz))
    egn_vec = np.zeros((n_kp, dynam_sz, dynam_sz), dtype=complex)
    omega2THz = 1 / (2 * np.pi * 1e12)

    for i in range(n_kp):
        inlpEig = dynam[i]
        E, V = eig(inlpEig)
        E = -np.real(np.diag(E))
        order = np.argsort(E)
        E = E[order]
        V = V[:, order]
        freq[i] = np.sqrt(E) * omega2THz
        egn_vec[i] = V

    freq = np.real(freq)

    # longitudinal acoustic mode
    frq_l = np.hstack([freq[:nshift, 2], freq[nshift:, 4]])

    # eigenvectors corresponding to the longitudinal mode
    egn_vec_l = np.vstack([egn_vec[:nshift, :, 2], egn_vec[nshift:, :, 4]])
    sgn_crtc_trm = np.sign(np.real(egn_vec_l[:, 2]))
    egn_vec_l = egn_vec_l.T * sgn_crtc_trm

    # create Gaussian wave packet
    for itr in range(len(idx_ko_matrix)):
        idx_ko = 5 * idx_ko_matrix[itr]
        ko = kp_norm[idx_ko]
        brding = 0.005
        amp = amp_matrix[itr]

        gauss = amp / (np.sqrt(2 * np.pi) * brding) * \
                np.exp(-((kp_norm - ko) / (np.sqrt(2) * brding)) ** 2)

        plt.figure()
        plt.plot(kp_norm, gauss, linewidth=2.2)
        plt.show()

        # wavepacket calculations
        n_cell = 10 * 10 * 390
        p_si = p[:n_cell * 8]
        lp_si = p_si[::8, 2:5] - p_si[0, 2:5]

        atm_site = np.hstack([np.tile(lp_si[:, i], n_atm_uitcel).reshape(-1, 1) for i in range(3)])

        crct_trm = np.exp(-1j * np.angle(egn_vec_l[2, idx_ko]))

        u = np.zeros((len(atm_site), 3))

        for i in range(3):
            u[:, i] = np.real(gauss[idx_ko] * np.tile(egn_vec_l[i::3, idx_ko], n_cell) *
                              np.exp(-1j * np.dot(atm_site, kp[:, idx_ko])) * crct_trm)

    return freq, egn_vec_l

def wavepacket_simulator_tr(n_kp, lc, n_cel, M, pos, idx_ko_matrix, amp_matrix, nshift):
    """
    Phonon dispersion and wavepacket analysis.
    
    Parameters:
    -----------
    n_kp: int
        Number of k-points.
    lc: float
        Lattice constant.
    n_cel: int
        Number of cells.
    M: np.array
        Noisy Hessian matrix.
    pos: np.array
        Atom positions.
    idx_ko_matrix: list
        Index for k-space local centers.
    amp_matrix: list
        Amplitude for each k-space.
    nshift: int
        Shift for central unit cell index.
    """

    # symmetrize Hessian matrix
    H = (np.triu(M) + np.tril(M.T)) / 2  # symmetric Hessian matrix
    Hsn = np.triu(H) + np.triu(H, 1).T  # Hessian matrix

    # k-points
    kp = np.vstack([np.zeros((2, n_kp)), np.linspace(0, np.pi / lc, n_kp)])  # wavevector
    kp_norm = np.linalg.norm(kp, axis=0)

    # sort and extract lattice positions
    pos_sorted = pos[pos[:, 0].argsort()]  # sort rows by atom ID
    n_atm_uitcel = 8
    lp = pos_sorted[::n_atm_uitcel, 2:5]

    # central cell and relative lattice points
    ctr_cell = int(np.floor(n_cel**3 / 2)) + nshift  # central unit cell (modified by nshift)
    r = lp - lp[ctr_cell - 1, :]  # lattice points w.r.t. central cell

    # dynamical matrix size and initialization
    dynam_sz = 3 * n_atm_uitcel
    dynam = np.zeros((n_kp, dynam_sz, dynam_sz), dtype=complex)
    inloop_var = np.zeros((dynam_sz, dynam_sz), dtype=complex)

    # build dynamical matrix
    for i in range(n_kp):
        for ii in range(r.shape[0]):
            inlpvar = Hsn[dynam_sz * ii : dynam_sz * (ii + 1), (ctr_cell - 1) * dynam_sz : ctr_cell * dynam_sz] * \
                      np.exp(-1j * np.dot(r[ii, :], kp[:, i]))
            inloop_var += inlpvar
        dynam[i, :, :] = inloop_var
        inloop_var.fill(0)  # reset for the next iteration

    # eigenvalue and eigenvector calculation
    freq = np.zeros((n_kp, dynam_sz))
    egn_vec = np.zeros((n_kp, dynam_sz, dynam_sz), dtype=complex)
    omega2THz = 1 / (2 * np.pi * 1e12)

    for i in range(n_kp):
        inlpEig = dynam[i, :, :]
        E, V = np.linalg.eigh(inlpEig)
        E = -np.real(E)
        order = np.argsort(E)
        E = E[order]
        V = V[:, order]
        freq[i, :] = np.sqrt(E) * omega2THz
        egn_vec[i, :, :] = V

    # band modification
    dir_idx = 1  # direction index (1 = x, 2 = y, 3 = z)
    frq_t = freq[:, 0]

    plt.figure()
    plt.plot(kp[2, :], frq_t, linewidth=2.2)
    plt.show()

    # eigenvector transformations
    egn_vec_t1 = egn_vec[:, :, 0].T
    egn_vec_t2 = egn_vec[:, :, 1].T

    alpha = np.arctan2(np.sum(-egn_vec_t1 * np.tile([0, 1, 0], (8, 1))), np.sum(egn_vec_t2 * np.tile([0, 1, 0], (8, 1))))
    beta = np.arctan2(np.sum(-egn_vec_t1 * np.tile([1, 0, 0], (8, 1))), np.sum(egn_vec_t2 * np.tile([1, 0, 0], (8, 1))))
    alpha[np.isnan(alpha)] = np.pi / 2
    beta[np.isnan(beta)] = np.pi / 2

    egn_vec_x = egn_vec_t1 * np.cos(alpha) + egn_vec_t2 * np.sin(alpha)
    egn_vec_y = egn_vec_t1 * np.cos(beta) + egn_vec_t2 * np.sin(beta)

    sgn_crtc_trm = np.sign(np.real(egn_vec_x[dir_idx, :]))
    egn_vec_t = egn_vec_x * sgn_crtc_trm

    # Gaussian distribution and wave packet analysis
    for itr in range(1):
        idx_ko = 5 * idx_ko_matrix[itr]
        ko = kp_norm[idx_ko]  # K-space local center
        brding = 0.005  # breading factor
        amp = amp_matrix[itr]  # wave amplitude
        gauss = np.zeros(n_kp)
        
        # gaussian distribution
        for i in range(n_kp):
            gauss[i] = amp / (np.sqrt(2 * np.pi) * brding) * np.exp(-((kp_norm[i] - ko) / (np.sqrt(2) * brding))**2)
        
    
    return freq, egn_vec, gauss
