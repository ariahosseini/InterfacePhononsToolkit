"""
A Monte Carlo ray tracing model that simulate the auto-correlation heat current process
 in materials with nanoscale porosity
"""

from scipy import integrate
import numpy as np
from .autocorrelation_func import acf, acf_fft


time_index = lambda time, max_time, time_steps: int(round((time_steps-1)*time/max_time))


def random_direction():

    # A method to pick a random direction for phonon propagation

    theta = np.arccos(1 - (2*np.random.uniform(0, 1)))  # theta is the angle with x in Cartesian coordinates
    phi = np.random.uniform(-np.pi, np.pi)              # phi is the angle with z in Cartesian coordinates

    return theta, phi


def get_random_phonon_spec_wall(tau_max: float, time_intervals: int,
                                v_g: float, tau_mean: float, L: float, P_reflection: float):

    """
    Compute auto-correlation for phonons between walls with spectral scattering

    :arg
        tau_max: float
            Maximum correlation time to compute the ACF out to
        time_intervals: int
            Time intervals
        v_g: float
            Phonon group velocity (magnitude)
        tau_mean: float
            Phonon's average lifetime
        L: float
            Spacing between walls
        P_reflection: float
            The probability of reflection
    :returns
        acf_func: np.ndarray
            Auto-correlation
        Int: np.ndarray
            Intensity
    """

    J = np.zeros([3, time_intervals])  # Initiate heat flux vector
    time = np.linspace(0, tau_max, time_intervals)  # Time
    theta, phi = random_direction()  # Get random direction

    # Phonon group velocity vector
    nv = np.array([np.cos(theta), np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi)])
    v = v_g * nv

    direction = float(np.sign(v[0]))  # Find the direction along x
    x = np.random.uniform(0, L)  # Get random starting position
    tau_hit = (L * 0.5 * (1 + direction) - x) / v[0]  # Time of flight to the wall
    hit_index = time_index(tau_hit, tau_max, time_intervals)  # Time index of the phonon flight to the wall
    delta_tau = L/np.abs(v[0])  # Time of flight between walls

    # Generate random lifetime
    r_tau = np.random.random()  # Random number between 0 and 1
    tau_o = -tau_mean * np.log(1 - r_tau)  # Phonon annihilation time

    # Find annihilation time index
    die_index = min(time_index(tau_o, tau_max, time_intervals), time_intervals)

    now_index = 0
    hit_count = 0  # counters

    while hit_index < die_index:

        row = np.ones(hit_index-now_index)
        J[:, now_index:hit_index] = np.array([v[i]*row for i in range(3)])
        hit_count += 1

        dice = np.random.random()
        if dice < P_reflection:
            direction *= -1.0
            v[0] *= -1.0

        tau_hit += delta_tau
        now_index = hit_index

        hit_index = time_index(tau_hit, tau_max, time_intervals)

    row = np.ones(die_index-now_index)
    J[:, now_index:die_index] = np.array([v[i]*row for i in range(3)])  # Heat flux
    acf_func = acf(time, J)  # Auto-correlation
    Int = np.array([integrate.cumtrapz(acf_func[i], time, initial = 0) for i in range(3)])  # Intensity
    return acf_func, Int


def get_random_phonon_spec_wall_fft(tau_max: float, time_intervals: int,
                                    v_g: float, tau_mean: float, L: float, P_reflection: float):

    """
    Compute auto-correlation for phonons between walls with spectral scattering using discrete fourier method

    :arg
        tau_max: float
            Maximum correlation time to compute the ACF out to
        time_intervals: int
            Time intervals
        v_g: float
            Phonon group velocity (magnitude)
        tau_mean: float
            Phonon's average lifetime
        L: float
            Spacing between walls
        P_reflection: float
            The probability of reflection
    :returns
        acf_fft: np.ndarray
            Auto-correlation
        Int_fft: np.ndarray
            Intensity
    """

    J = np.zeros([3, time_intervals])  # Initiate heat flux vector
    time = np.linspace(0, tau_max, time_intervals)  # Time
    theta, phi = random_direction()  # Get random direction

    # Phonon group velocity vector
    nv = np.array([np.cos(theta), np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi)])
    v = v_g * nv
    direction = float(np.sign(v[0]))  # Find the direction along x

    # Get random starting position and time of flight to the wall
    x = np.random.uniform(0, L)  # Get random starting position

    tau_hit = (L * 0.5 *(1 + direction) - x) / v[0]
    hit_index = time_index(tau_hit, tau_max, time_intervals)  # Time index of the phonon flight to the wall

    delta_tau = L/np.abs(v[0])  # Time of flight between walls

    r_tau = np.random.random()  # Random number between 0 and 1
    tau_o = -tau_mean * np.log(1 - r_tau)  # Phonon annihilation time
    # Find annihilation time index
    die_index = min(time_index(tau_o, tau_max, time_intervals), time_intervals)

    now_index = 0
    hit_count = 0

    while hit_index < die_index:

        row = np.ones(hit_index-now_index)
        J[:, now_index:hit_index] = np.array([v[i]*row for i in range(3)])
        hit_count += 1

        dice = np.random.random()
        if dice < P_reflection:

            direction *= -1.0
            v[0] *= -1.0

        tau_hit += delta_tau
        now_index = hit_index

        hit_index = time_index(tau_hit, tau_max, time_intervals)

    row = np.ones(die_index-now_index)
    J[:, now_index:die_index] = np.array([v[i]*row for i in range(3)])  # Heat flux
    acf_fft_func = acf_fft(time, J)  # Auto-correlation
    time_ind = (1+(time_intervals-1)/2)
    # Intensity
    Int_fft = np.array([integrate.cumtrapz(acf_fft_func[i, :time_ind], time[:time_ind], initial = 0) for i in range(3)])
    return acf_fft_func, Int_fft


def get_random_phonon_diff_wall(tau_max: float, time_intervals: int,
                                    v_g: float, tau_mean: float, L: float, P_reflection: float):

    """
    compute auto-correlation for phonons between walls with diffusive scatting

    :arg
        tau_max: float
            Maximum correlation time to compute the ACF out to
        time_intervals: int
            Time intervals
        v_g: float
            Phonon group velocity (magnitude)
        tau_mean: float
            Phonon's average lifetime
        L: float
            Spacing between walls
        P_reflection: float
            The probability of reflection
    :returns
        acf_func: np.ndarray
            Auto-correlation
        Int: np.ndarray
            Intensity
    """

    J = np.zeros([3, time_intervals])  # Initiate heat flux vector
    time = np.linspace(0, tau_max, time_intervals)  # Time

    # Get random direction
    theta, phi = random_direction()

    # Phonon group velocity vector
    nv = np.array([np.cos(theta), np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi)])
    v = v_g * nv
    direction = float(np.sign(v[0]))  # Find the direction along x

    x = np.random.uniform(0, L)  # Get random starting position and time of flight to the wall
    tau_hit = (L * 0.5 *(1 + direction) - x) / v[0]  # Time of flight to the wall
    delta_tau = L/np.abs(v[0])  # Time of flight between walls

    r_tau = np.random.random()  # Random number between 0 and 1
    tau_o = -tau_mean * np.log(1 - r_tau)  # Phonon annihilation time

    tau_o_ind = int(round((time_intervals-1)*tau_o/tau_max))
    max_tau_ind = min(tau_o_ind, time_intervals)
    for i in range(max_tau_ind):
        if time[i] >= tau_hit:
            dice = np.random.random()
            if dice < P_reflection:
                direction *= -1.0
                theta, phi = random_direction()
                nv = np.array([np.cos(theta), np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi)])
                nv[0] = abs(nv[0])*direction
                v = v_g * nv
                delta_tau = L/np.abs(v[0])  # Time of flight between walls
            tau_hit += delta_tau
        J[:, i] = v  # Heat flux

    acf_func = acf(time, J)  # Auto-correlation
    # Intensity
    Int = np.array([integrate.cumtrapz(acf_func[i], time, initial = 0) for i in range(3)])
    return acf_func, Int


def random_beta_angle(beta_min: float, beta_max: float):

    return np.arcsin(np.random.random() * (np.sin(beta_max) - np.sin(beta_min))+np.sin(beta_min))


def PT(theta_prime: float, alpha: float):

    return (np.abs(np.cos(theta_prime)) - alpha)/np.abs(np.cos(theta_prime))


def get_random_phonon_spec_cyli(tau_max: float, time_intervals: int,
                                    v_g: float, tau_mean: float, L: float, alpha: float):

    """
    A method to compute auto-correlation for phonons between cylindrical pores with spectral scattering

    :arg
        tau_max: float
            Maximum correlation time to compute the ACF out to
        time_intervals: int
            Time intervals
        v_g: float
            Phonon group velocity (magnitude)
        tau_mean: float
            Phonon's average lifetime
        L: float
            Spacing between walls
        alpha: float
            The probability of reflection
    :returns
        acf_func: np.ndarray
            Auto-correlation
        Int: np.ndarray
            Intensity
    """

    J = np.zeros([3, time_intervals])  # Initiate heat flux vector
    time = np.linspace(0, tau_max, time_intervals)  # Time

    theta, phi = random_direction()  # Get random direction

    # Phonon group velocity vector
    nv = np.array([np.cos(theta), np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi)])
    v = v_g * nv

    direction = float(np.sign(v[0]))  # Find the direction along x
    vxz = np.sqrt((v[0])**2 + (v[2])**2)  # Phonon group velocity vector
    theta_prime = np.arctan2(v[2], v[0])  # Phonon group velocity vector angel

    # Get random starting position and time of flight to the wall
    x = np.random.uniform(0, L)  # Get random starting position
    tau_hit = (L * 0.5 *(1 + direction) - x) / v[0]  # Time of flight to the wall
    delta_tau = L/np.abs(v[0])  # Time of flight between walls

    # Get random lifetime
    r_tau = np.random.random()  # Random number between 0 and 1
    tau_o = -tau_mean * np.log(1 - r_tau)  # Phonon annihilation time

    reflection = 0
    transmitted = 0
    tau_o_ind = int(round((time_intervals-1)*tau_o/tau_max))
    max_tau_ind = min(tau_o_ind, time_intervals)

    for i in range(max_tau_ind):

        if time[i] >= tau_hit:

            T = PT(theta_prime, alpha)
            beta_min = -np.pi/2
            beta_max = np.pi/2
            if T < 0:           # We have shadowing and so need to adjust the limits on the
                # possible incidence angles Beta
                if np.sin(theta_prime) * np.cos(theta_prime) >= 0:
                    beta_min = np.arcsin(1 - (2 * np.abs(np.cos(theta_prime)))/alpha)
                else:
                    beta_max = -np.arcsin(1 - (2 * np.abs(np.cos(theta_prime)))/alpha)
                beta = random_beta_angle(beta_min, beta_max)
                reflection += 1
            else:
                if np.random.random() < T: # Transmitted
                    beta = np.pi/2
                    transmitted += 1
                else:  # Scattered
                    beta = random_beta_angle(beta_min, beta_max)
                    reflection += 1
            theta_prime += np.pi - (2*beta)  # New direction of travel in the x-z plane
            v[0] = vxz * np.cos(theta_prime)
            v[2] = vxz * np.sin(theta_prime)
            tau_hit += L/np.abs(v[0])
        J[:, i] = v  # Heat flux

    acf_func = acf(time, J)  # Auto-correlation
    Int = np.array([integrate.cumtrapz(acf_func[i], time, initial = 0) for i in range(3)])   # Intensity

    return acf_func, Int


def get_random_phonon_diff_cyli(tau_max: float, time_intervals: int,
                                    v_g: float, tau_mean: float, L: float, alpha: float):

    """
    compute auto-correlation for phonons between cylindrical pores with diffusive scattering

    :arg
        tau_max: float
            Maximum correlation time to compute the ACF out to
        time_intervals: int
            Time intervals
        v_g: float
            Phonon group velocity (magnitude)
        tau_mean: float
            Phonon's average lifetime
        L: float
            Spacing between walls
        alpha: float
            The probability of reflection
    :returns
        acf_func: np.ndarray
            Auto-correlation
        Int: np.ndarray
            Intensity
    """

    J = np.zeros([3, time_intervals])  # Initiate heat flux vector
    time = np.linspace(0, tau_max, time_intervals)  # Time

    theta, phi = random_direction()  # Get random direction

    # Phonon group velocity vector
    nv = np.array([np.cos(theta), np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi)])
    v = v_g * nv
    direction = float(np.sign(v[0]))  # Find the direction along x
    vxz = np.sqrt((v[0])**2 + (v[2])**2)  # Phonon group velocity vector
    theta_prime = np.arctan2(v[2], v[0])  # Phonon group velocity vector angel

    # Get random starting position and time of flight to the wall
    x = np.random.uniform(0, L)  # Get random starting position
    tau_hit = (L * 0.5 *(1 + direction) - x) / v[0]  # Time of flight to the wall
    delta_tau = L/np.abs(v[0])  # Time of flight between walls

    # Get random lifetime
    r_tau = np.random.random()  # Random number between 0 and 1
    tau_o = -tau_mean * np.log(1 - r_tau) # Phonon annihilation time

    reflection = 0
    transmitted = 0

    # Find annihilation time index
    tau_o_ind = int(round((time_intervals-1)*tau_o/tau_max))
    max_tau_ind = min(tau_o_ind, time_intervals)

    for i in range(max_tau_ind):

        if time[i] >= tau_hit:

            T = PT(theta_prime, alpha)
            beta_min = -np.pi/2
            beta_max = np.pi/2
            if T < 0:       # We have shadowing and so need to adjust the limits on the
                            # possible incidence angles Beta
                if np.sin(theta_prime) * np.cos(theta_prime) >= 0:
                    beta_min = np.arcsin(1 - (2 * np.abs(np.cos(theta_prime)))/alpha)
                else:
                    beta_max = -np.arcsin(1 - (2 * np.abs(np.cos(theta_prime)))/alpha)
                beta = random_beta_angle(beta_min, beta_max)
                reflection += 1
            else:
                if np.random.random() < T:  # Transmitted
                    beta = np.pi/2
                    transmitted += 1
                else:  # Scattered
                    beta = random_beta_angle(beta_min, beta_max)
                    reflection += 1
            theta_prime += np.pi - (2*beta)  # New direction of travel in the x-z plane
            v[0] = vxz * np.cos(theta_prime)
            v[2] = vxz * np.sin(theta_prime)
            tau_hit += L/np.abs(v[0])
        J[:,i] = v  # Heat flux

    acf_func = acf(time, J)  # Auto-correlation
    # Intensity
    Int = np.array([integrate.cumtrapz(acf_func[i], time, initial = 0) for i in range(3)])

    return acf_func, Int


def average_phonons(num_blocks: int, samples_per_block: int, tau_max: float,
                    time_intervals: int, v_g: float, tau_mean: float, L: float, alpha: float, mode: str):

    """
    Average over num_blocks of samples_per_block

    :arg
        num_blocks: int
            Number of blocks to average
        samples_per_block: int
            Number of samples in each block
        tau_max: float
            Maximum correlation time to compute the ACF out to
        time_intervals: int
            Time intervals
        v_g: float
            Phonon group velocity (magnitude)
        tau_mean: float
            Phonon's average lifetime
        L: float
            Spacing between walls
        alpha: float
            The probability of reflection
        mode: str
            Scattering type — SpecWall or SpecWall_fft, DiffWall, SpecCyli, DiffCyli
    :returns
        ACFm: np.ndarray
            Mean of auto-correlation
        ACFe: np.ndarray
            Standard deviation of auto-correlation
        km: np.ndarray
            Mean of thermal conductivity
        ke : np.ndarray
            Standard deviation of thermal conductivity
    """

    ACF = np.zeros([3, num_blocks, time_intervals])
    kappa = np.zeros([3, num_blocks, time_intervals])

    for i in range(num_blocks):

        for j in range(samples_per_block):

            if mode == "SpecWall":
                acf_func, Int = get_random_phonon_spec_wall(tau_max, time_intervals, v_g, tau_mean, L, alpha)
            if mode == "SpecWall_fft":
                acf_func, Int = get_random_phonon_spec_wall(tau_max, time_intervals, v_g, tau_mean, L, alpha)
            if mode == "DiffWall":
                acf_func, Int = get_random_phonon_diff_wall(tau_max, time_intervals, v_g, tau_mean, L, alpha)
            if mode == "SpecCyli":
                acf_func, Int = get_random_phonon_spec_cyli(tau_max, time_intervals, v_g, tau_mean, L, alpha)
            if mode == "DiffCyli":
                acf_func, Int = get_random_phonon_diff_cyli(tau_max, time_intervals, v_g, tau_mean, L, alpha)

            ACF[:, i] += acf_func
            kappa[:, i] += Int

    ACF /= tau_mean*float(samples_per_block)
    kappa /= tau_mean*float(samples_per_block)

    ACFm = ACF.mean(axis=1)  # Mean of auto-correlation
    ACFe = ACF.std(axis=1)/np.sqrt(float(num_blocks-1))  # Standard deviation of auto-correlation
    km = kappa.mean(axis=1)  # Mean of thermal conductivity
    ke = kappa.std(axis=1)/np.sqrt(float(num_blocks-1))  # Standard deviation of thermal conductivity

    return ACFm, ACFe, km, ke"""
A Monte Carlo ray tracing model that simulate the auto-correlation heat current process
 in materials with nanoscale porosity
"""

from scipy import integrate
import numpy as np
from .autocorrelation_func import acf, acf_fft


time_index = lambda time, max_time, time_steps: int(round((time_steps-1)*time/max_time))


def random_direction():

    # A method to pick a random direction for phonon propagation

    theta = np.arccos(1 - (2*np.random.uniform(0, 1)))  # theta is the angle with x in Cartesian coordinates
    phi = np.random.uniform(-np.pi, np.pi)              # phi is the angle with z in Cartesian coordinates

    return theta, phi


def get_random_phonon_spec_wall(tau_max: float, time_intervals: int,
                                v_g: float, tau_mean: float, L: float, P_reflection: float):

    """
    Compute auto-correlation for phonons between walls with spectral scattering

    :arg
        tau_max: float
            Maximum correlation time to compute the ACF out to
        time_intervals: int
            Time intervals
        v_g: float
            Phonon group velocity (magnitude)
        tau_mean: float
            Phonon's average lifetime
        L: float
            Spacing between walls
        P_reflection: float
            The probability of reflection
    :returns
        acf_func: np.ndarray
            Auto-correlation
        Int: np.ndarray
            Intensity
    """

    J = np.zeros([3, time_intervals])  # Initiate heat flux vector
    time = np.linspace(0, tau_max, time_intervals)  # Time
    theta, phi = random_direction()  # Get random direction

    # Phonon group velocity vector
    nv = np.array([np.cos(theta), np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi)])
    v = v_g * nv

    direction = float(np.sign(v[0]))  # Find the direction along x
    x = np.random.uniform(0, L)  # Get random starting position
    tau_hit = (L * 0.5 * (1 + direction) - x) / v[0]  # Time of flight to the wall
    hit_index = time_index(tau_hit, tau_max, time_intervals)  # Time index of the phonon flight to the wall
    delta_tau = L/np.abs(v[0])  # Time of flight between walls

    # Generate random lifetime
    r_tau = np.random.random()  # Random number between 0 and 1
    tau_o = -tau_mean * np.log(1 - r_tau)  # Phonon annihilation time

    # Find annihilation time index
    die_index = min(time_index(tau_o, tau_max, time_intervals), time_intervals)

    now_index = 0
    hit_count = 0  # counters

    while hit_index < die_index:

        row = np.ones(hit_index-now_index)
        J[:, now_index:hit_index] = np.array([v[i]*row for i in range(3)])
        hit_count += 1

        dice = np.random.random()
        if dice < P_reflection:
            direction *= -1.0
            v[0] *= -1.0

        tau_hit += delta_tau
        now_index = hit_index

        hit_index = time_index(tau_hit, tau_max, time_intervals)

    row = np.ones(die_index-now_index)
    J[:, now_index:die_index] = np.array([v[i]*row for i in range(3)])  # Heat flux
    acf_func = acf(time, J)  # Auto-correlation
    Int = np.array([integrate.cumtrapz(acf_func[i], time, initial = 0) for i in range(3)])  # Intensity
    return acf_func, Int


def get_random_phonon_spec_wall_fft(tau_max: float, time_intervals: int,
                                    v_g: float, tau_mean: float, L: float, P_reflection: float):

    """
    Compute auto-correlation for phonons between walls with spectral scattering using discrete fourier method

    :arg
        tau_max: float
            Maximum correlation time to compute the ACF out to
        time_intervals: int
            Time intervals
        v_g: float
            Phonon group velocity (magnitude)
        tau_mean: float
            Phonon's average lifetime
        L: float
            Spacing between walls
        P_reflection: float
            The probability of reflection
    :returns
        acf_fft: np.ndarray
            Auto-correlation
        Int_fft: np.ndarray
            Intensity
    """

    J = np.zeros([3, time_intervals])  # Initiate heat flux vector
    time = np.linspace(0, tau_max, time_intervals)  # Time
    theta, phi = random_direction()  # Get random direction

    # Phonon group velocity vector
    nv = np.array([np.cos(theta), np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi)])
    v = v_g * nv
    direction = float(np.sign(v[0]))  # Find the direction along x

    # Get random starting position and time of flight to the wall
    x = np.random.uniform(0, L)  # Get random starting position

    tau_hit = (L * 0.5 *(1 + direction) - x) / v[0]
    hit_index = time_index(tau_hit, tau_max, time_intervals)  # Time index of the phonon flight to the wall

    delta_tau = L/np.abs(v[0])  # Time of flight between walls

    r_tau = np.random.random()  # Random number between 0 and 1
    tau_o = -tau_mean * np.log(1 - r_tau)  # Phonon annihilation time
    # Find annihilation time index
    die_index = min(time_index(tau_o, tau_max, time_intervals), time_intervals)

    now_index = 0
    hit_count = 0

    while hit_index < die_index:

        row = np.ones(hit_index-now_index)
        J[:, now_index:hit_index] = np.array([v[i]*row for i in range(3)])
        hit_count += 1

        dice = np.random.random()
        if dice < P_reflection:

            direction *= -1.0
            v[0] *= -1.0

        tau_hit += delta_tau
        now_index = hit_index

        hit_index = time_index(tau_hit, tau_max, time_intervals)

    row = np.ones(die_index-now_index)
    J[:, now_index:die_index] = np.array([v[i]*row for i in range(3)])  # Heat flux
    acf_fft_func = acf_fft(time, J)  # Auto-correlation
    time_ind = (1+(time_intervals-1)/2)
    # Intensity
    Int_fft = np.array([integrate.cumtrapz(acf_fft_func[i, :time_ind], time[:time_ind], initial = 0) for i in range(3)])
    return acf_fft_func, Int_fft


def get_random_phonon_diff_wall(tau_max: float, time_intervals: int,
                                    v_g: float, tau_mean: float, L: float, P_reflection: float):

    """
    compute auto-correlation for phonons between walls with diffusive scatting

    :arg
        tau_max: float
            Maximum correlation time to compute the ACF out to
        time_intervals: int
            Time intervals
        v_g: float
            Phonon group velocity (magnitude)
        tau_mean: float
            Phonon's average lifetime
        L: float
            Spacing between walls
        P_reflection: float
            The probability of reflection
    :returns
        acf_func: np.ndarray
            Auto-correlation
        Int: np.ndarray
            Intensity
    """

    J = np.zeros([3, time_intervals])  # Initiate heat flux vector
    time = np.linspace(0, tau_max, time_intervals)  # Time

    # Get random direction
    theta, phi = random_direction()

    # Phonon group velocity vector
    nv = np.array([np.cos(theta), np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi)])
    v = v_g * nv
    direction = float(np.sign(v[0]))  # Find the direction along x

    x = np.random.uniform(0, L)  # Get random starting position and time of flight to the wall
    tau_hit = (L * 0.5 *(1 + direction) - x) / v[0]  # Time of flight to the wall
    delta_tau = L/np.abs(v[0])  # Time of flight between walls

    r_tau = np.random.random()  # Random number between 0 and 1
    tau_o = -tau_mean * np.log(1 - r_tau)  # Phonon annihilation time

    tau_o_ind = int(round((time_intervals-1)*tau_o/tau_max))
    max_tau_ind = min(tau_o_ind, time_intervals)
    for i in range(max_tau_ind):
        if time[i] >= tau_hit:
            dice = np.random.random()
            if dice < P_reflection:
                direction *= -1.0
                theta, phi = random_direction()
                nv = np.array([np.cos(theta), np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi)])
                nv[0] = abs(nv[0])*direction
                v = v_g * nv
                delta_tau = L/np.abs(v[0])  # Time of flight between walls
            tau_hit += delta_tau
        J[:, i] = v  # Heat flux

    acf_func = acf(time, J)  # Auto-correlation
    # Intensity
    Int = np.array([integrate.cumtrapz(acf_func[i], time, initial = 0) for i in range(3)])
    return acf_func, Int


def random_beta_angle(beta_min: float, beta_max: float):

    return np.arcsin(np.random.random() * (np.sin(beta_max) - np.sin(beta_min))+np.sin(beta_min))


def PT(theta_prime: float, alpha: float):

    return (np.abs(np.cos(theta_prime)) - alpha)/np.abs(np.cos(theta_prime))


def get_random_phonon_spec_cyli(tau_max: float, time_intervals: int,
                                    v_g: float, tau_mean: float, L: float, alpha: float):

    """
    A method to compute auto-correlation for phonons between cylindrical pores with spectral scattering

    :arg
        tau_max: float
            Maximum correlation time to compute the ACF out to
        time_intervals: int
            Time intervals
        v_g: float
            Phonon group velocity (magnitude)
        tau_mean: float
            Phonon's average lifetime
        L: float
            Spacing between walls
        alpha: float
            The probability of reflection
    :returns
        acf_func: np.ndarray
            Auto-correlation
        Int: np.ndarray
            Intensity
    """

    J = np.zeros([3, time_intervals])  # Initiate heat flux vector
    time = np.linspace(0, tau_max, time_intervals)  # Time

    theta, phi = random_direction()  # Get random direction

    # Phonon group velocity vector
    nv = np.array([np.cos(theta), np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi)])
    v = v_g * nv

    direction = float(np.sign(v[0]))  # Find the direction along x
    vxz = np.sqrt((v[0])**2 + (v[2])**2)  # Phonon group velocity vector
    theta_prime = np.arctan2(v[2], v[0])  # Phonon group velocity vector angel

    # Get random starting position and time of flight to the wall
    x = np.random.uniform(0, L)  # Get random starting position
    tau_hit = (L * 0.5 *(1 + direction) - x) / v[0]  # Time of flight to the wall
    delta_tau = L/np.abs(v[0])  # Time of flight between walls

    # Get random lifetime
    r_tau = np.random.random()  # Random number between 0 and 1
    tau_o = -tau_mean * np.log(1 - r_tau)  # Phonon annihilation time

    reflection = 0
    transmitted = 0
    tau_o_ind = int(round((time_intervals-1)*tau_o/tau_max))
    max_tau_ind = min(tau_o_ind, time_intervals)

    for i in range(max_tau_ind):

        if time[i] >= tau_hit:

            T = PT(theta_prime, alpha)
            beta_min = -np.pi/2
            beta_max = np.pi/2
            if T < 0:           # We have shadowing and so need to adjust the limits on the
                # possible incidence angles Beta
                if np.sin(theta_prime) * np.cos(theta_prime) >= 0:
                    beta_min = np.arcsin(1 - (2 * np.abs(np.cos(theta_prime)))/alpha)
                else:
                    beta_max = -np.arcsin(1 - (2 * np.abs(np.cos(theta_prime)))/alpha)
                beta = random_beta_angle(beta_min, beta_max)
                reflection += 1
            else:
                if np.random.random() < T: # Transmitted
                    beta = np.pi/2
                    transmitted += 1
                else:  # Scattered
                    beta = random_beta_angle(beta_min, beta_max)
                    reflection += 1
            theta_prime += np.pi - (2*beta)  # New direction of travel in the x-z plane
            v[0] = vxz * np.cos(theta_prime)
            v[2] = vxz * np.sin(theta_prime)
            tau_hit += L/np.abs(v[0])
        J[:, i] = v  # Heat flux

    acf_func = acf(time, J)  # Auto-correlation
    Int = np.array([integrate.cumtrapz(acf_func[i], time, initial = 0) for i in range(3)])   # Intensity

    return acf_func, Int


def get_random_phonon_diff_cyli(tau_max: float, time_intervals: int,
                                    v_g: float, tau_mean: float, L: float, alpha: float):

    """
    compute auto-correlation for phonons between cylindrical pores with diffusive scattering

    :arg
        tau_max: float
            Maximum correlation time to compute the ACF out to
        time_intervals: int
            Time intervals
        v_g: float
            Phonon group velocity (magnitude)
        tau_mean: float
            Phonon's average lifetime
        L: float
            Spacing between walls
        alpha: float
            The probability of reflection
    :returns
        acf_func: np.ndarray
            Auto-correlation
        Int: np.ndarray
            Intensity
    """

    J = np.zeros([3, time_intervals])  # Initiate heat flux vector
    time = np.linspace(0, tau_max, time_intervals)  # Time

    theta, phi = random_direction()  # Get random direction

    # Phonon group velocity vector
    nv = np.array([np.cos(theta), np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi)])
    v = v_g * nv
    direction = float(np.sign(v[0]))  # Find the direction along x
    vxz = np.sqrt((v[0])**2 + (v[2])**2)  # Phonon group velocity vector
    theta_prime = np.arctan2(v[2], v[0])  # Phonon group velocity vector angel

    # Get random starting position and time of flight to the wall
    x = np.random.uniform(0, L)  # Get random starting position
    tau_hit = (L * 0.5 *(1 + direction) - x) / v[0]  # Time of flight to the wall
    delta_tau = L/np.abs(v[0])  # Time of flight between walls

    # Get random lifetime
    r_tau = np.random.random()  # Random number between 0 and 1
    tau_o = -tau_mean * np.log(1 - r_tau) # Phonon annihilation time

    reflection = 0
    transmitted = 0

    # Find annihilation time index
    tau_o_ind = int(round((time_intervals-1)*tau_o/tau_max))
    max_tau_ind = min(tau_o_ind, time_intervals)

    for i in range(max_tau_ind):

        if time[i] >= tau_hit:

            T = PT(theta_prime, alpha)
            beta_min = -np.pi/2
            beta_max = np.pi/2
            if T < 0:       # We have shadowing and so need to adjust the limits on the
                            # possible incidence angles Beta
                if np.sin(theta_prime) * np.cos(theta_prime) >= 0:
                    beta_min = np.arcsin(1 - (2 * np.abs(np.cos(theta_prime)))/alpha)
                else:
                    beta_max = -np.arcsin(1 - (2 * np.abs(np.cos(theta_prime)))/alpha)
                beta = random_beta_angle(beta_min, beta_max)
                reflection += 1
            else:
                if np.random.random() < T:  # Transmitted
                    beta = np.pi/2
                    transmitted += 1
                else:  # Scattered
                    beta = random_beta_angle(beta_min, beta_max)
                    reflection += 1
            theta_prime += np.pi - (2*beta)  # New direction of travel in the x-z plane
            v[0] = vxz * np.cos(theta_prime)
            v[2] = vxz * np.sin(theta_prime)
            tau_hit += L/np.abs(v[0])
        J[:,i] = v  # Heat flux

    acf_func = acf(time, J)  # Auto-correlation
    # Intensity
    Int = np.array([integrate.cumtrapz(acf_func[i], time, initial = 0) for i in range(3)])

    return acf_func, Int


def average_phonons(num_blocks: int, samples_per_block: int, tau_max: float,
                    time_intervals: int, v_g: float, tau_mean: float, L: float, alpha: float, mode: str):

    """
    Average over num_blocks of samples_per_block

    :arg
        num_blocks: int
            Number of blocks to average
        samples_per_block: int
            Number of samples in each block
        tau_max: float
            Maximum correlation time to compute the ACF out to
        time_intervals: int
            Time intervals
        v_g: float
            Phonon group velocity (magnitude)
        tau_mean: float
            Phonon's average lifetime
        L: float
            Spacing between walls
        alpha: float
            The probability of reflection
        mode: str
            Scattering type — SpecWall or SpecWall_fft, DiffWall, SpecCyli, DiffCyli
    :returns
        ACFm: np.ndarray
            Mean of auto-correlation
        ACFe: np.ndarray
            Standard deviation of auto-correlation
        km: np.ndarray
            Mean of thermal conductivity
        ke : np.ndarray
            Standard deviation of thermal conductivity
    """

    ACF = np.zeros([3, num_blocks, time_intervals])
    kappa = np.zeros([3, num_blocks, time_intervals])

    for i in range(num_blocks):

        for j in range(samples_per_block):

            if mode == "SpecWall":
                acf_func, Int = get_random_phonon_spec_wall(tau_max, time_intervals, v_g, tau_mean, L, alpha)
            if mode == "SpecWall_fft":
                acf_func, Int = get_random_phonon_spec_wall(tau_max, time_intervals, v_g, tau_mean, L, alpha)
            if mode == "DiffWall":
                acf_func, Int = get_random_phonon_diff_wall(tau_max, time_intervals, v_g, tau_mean, L, alpha)
            if mode == "SpecCyli":
                acf_func, Int = get_random_phonon_spec_cyli(tau_max, time_intervals, v_g, tau_mean, L, alpha)
            if mode == "DiffCyli":
                acf_func, Int = get_random_phonon_diff_cyli(tau_max, time_intervals, v_g, tau_mean, L, alpha)

            ACF[:, i] += acf_func
            kappa[:, i] += Int

    ACF /= tau_mean*float(samples_per_block)
    kappa /= tau_mean*float(samples_per_block)

    ACFm = ACF.mean(axis=1)  # Mean of auto-correlation
    ACFe = ACF.std(axis=1)/np.sqrt(float(num_blocks-1))  # Standard deviation of auto-correlation
    km = kappa.mean(axis=1)  # Mean of thermal conductivity
    ke = kappa.std(axis=1)/np.sqrt(float(num_blocks-1))  # Standard deviation of thermal conductivity

    return ACFm, ACFe, km, ke
