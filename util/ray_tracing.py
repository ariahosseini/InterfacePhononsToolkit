"""

ray_tracing.py computes ...


Authors: S. Aria Hosseini and P.Alex Greaney
Email: shoss008@ucr.edu

"""


from scipy import integrate
from .autocorrelation_func import *

TimeIndex = lambda time, maxtime, timesteps: int(round((timesteps-1)*time/maxtime))

def RandomDirection():

    # A method to pick a random direction for phonon propagation

    theta = np.arccos(1 - (2*np.random.uniform(0, 1)))  # theta is the angle with x in Cartesian coordinates
    phi = np.random.uniform(-np.pi, np.pi)              # phi is the angle with z in Cartesian coordinates

    return theta, phi


def getRandomPhononSpecWall(tau_max, time_intervals, v_g, tau_mean, L, P_reflection):

    J = np.zeros([3, time_intervals])                   # Initiate heat flux vector, an array of 3 by time_intervals
    time = np.linspace(0, tau_max, time_intervals)      # Time
    theta, phi = RandomDirection()                      # Get random direction

    # Phonon group velocity vector
    nv = np.array([np.cos(theta), np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi)])
    v = v_g * nv
    direction = float(np.sign(v[0]))                    # Find the direction along x

    x = np.random.uniform(0, L)                         # Get random starting position
    tau_hit = (L * 0.5 * (1 + direction) - x) / v[0]   # Time of flight to the wall

    # Time index of the phonon flight to the wall
    hit_index = TimeIndex(tau_hit, tau_max, time_intervals)

    delta_tau = L/np.abs(v[0])                         # Time of flight between walls

    # Get random lifetime
    r_tau = np.random.random()
    tau_o = -tau_mean * np.log(1 - r_tau)
    die_index = min(TimeIndex(tau_o, tau_max, time_intervals), time_intervals)
    now_index = 0

    # counters
    hit_count = 0

    while hit_index < die_index:

        row = np.ones(hit_index-now_index)
        J[:, now_index:hit_index] = np.array([v[i]*row for i in range(3)])
        hit_count += 1

        dice = np.random.random()
        if dice < P_reflection:

            #       reflected += 1
            direction *= -1.0
            v[0] *= -1.0
#       else:
            #       transmitted += 1

        tau_hit += delta_tau
        now_index = hit_index

        hit_index = TimeIndex(tau_hit, tau_max, time_intervals)

    row = np.ones(die_index-now_index)
    J[:, now_index:die_index] = np.array([v[i]*row for i in range(3)])
    acf = ACF(time, J)
    I = np.array([integrate.cumtrapz(acf[i], time, initial = 0) for i in range(3)])
    return acf, I


def getRandomPhononSpecWall_fft(tau_max, time_intervals, v_g, tau_mean, L, P_reflection):

    J = np.zeros([3, time_intervals])

    time = np.linspace(0, tau_max, time_intervals)

    # Get random direction
    theta, phi = RandomDirection()
    nv = np.array([np.cos(theta), np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi)])
    v = v_g * nv
    direction = float(np.sign(v[0]))

    # Get random starting position and time of flight to the wall
    x = np.random.uniform(0, L)
    tau_hit = (L * 0.5 *(1 + direction) - x) / v[0]
    hit_index = TimeIndex(tau_hit, tau_max, time_intervals)

    delta_tau = L/np.abs(v[0])                      # Time of flight between walls

    # Get random lifetime
    r_tau = np.random.random()
    tau_o = -tau_mean * np.log(1 - r_tau)
    die_index = min(TimeIndex(tau_o, tau_max, time_intervals), time_intervals)
    now_index = 0

    # counters
    hit_count = 0

    while hit_index < die_index:

        row = np.ones(hit_index-now_index)
        J[:, now_index:hit_index] = np.array([v[i]*row for i in range(3)])
        hit_count += 1

        dice = np.random.random()
        if dice < P_reflection:
            #    reflected += 1
            direction *= -1.0
            v[0] *= -1.0
#       else:
            #    transmitted += 1

        tau_hit += delta_tau
        now_index = hit_index

        hit_index = TimeIndex(tau_hit, tau_max, time_intervals)

    row = np.ones(die_index-now_index)
    J[:, now_index:die_index] = np.array([v[i]*row for i in range(3)])
    cfft = ACF_FFT(time, J)
    time_ind = (1+(time_intervals-1)/2)
    Ifft = np.array([integrate.cumtrapz(cfft[i, :time_ind], time[:time_ind], initial = 0) for i in range(3)])
    return cfft, Ifft


def getRandomPhononDiffWall(tau_max, time_intervals, v_g, tau_0_0, d, P_reflection):

    J = np.zeros([3,time_intervals])
    t = np.linspace(0, tau_max, time_intervals)

    # Get random direction
    theta, phi = RandomDirection()
    nv = np.array([np.cos(theta), np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi)])
    v  = v_g * nv
    direction = float(np.sign(v[0]))

    # Get random starting position and time of flight to the wall
    x = np.random.uniform(0, d)
    tau_hit = (d * 0.5 *(1 + direction) - x) / v[0]
    Delta_tau = d/np.abs(v[0])                 # Time of flight between walls

    # Get random lifetime
    r_tau = np.random.random()
    tau_0 = -tau_0_0 * np.log(1 - r_tau)

    tau_0_ind = int(round((time_intervals-1)*tau_0/tau_max))
    max_tau_ind = min(tau_0_ind,time_intervals)
    for i in range(max_tau_ind):
        if t[i] >= tau_hit:
            dice = np.random.random()
            if dice < P_reflection:
                direction *= -1.0
                theta, phi = RandomDirection()
                nv = np.array([np.cos(theta), np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi)])
                nv[0] = abs(nv[0])*direction
                v  = v_g * nv
                Delta_tau = d/np.abs(v[0])     # Time of flight between walls
            tau_hit += Delta_tau
        J[:,i] = v

    c  = ACF(t, J)
    I   = np.array([integrate.cumtrapz(c[i], tau, initial = 0) for i in range(3)])
    return c, I


def randomBetaAngle(beta_min, beta_max):
    return np.arcsin(np.random.random() * (np.sin(beta_max) - np.sin(beta_min))+np.sin(beta_min))


def PT(theta_prime, alpha):
    return (np.abs(np.cos(theta_prime)) - alpha)/np.abs(np.cos(theta_prime))


def getRandomPhononSpecCyli(tau_max, time_intervals, v_g, tau_0_0, d, alpha):

    J = np.zeros([3,time_intervals])
    t = np.linspace(0, tau_max, time_intervals)

    # Get random direction
    theta, phi = RandomDirection()
    nv = np.array([np.cos(theta), np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi)])
    v  = v_g * nv
    direction = float(np.sign(v[0]))
    vxz = np.sqrt((v[0])**2 + (v[2])**2)
    theta_prime = np.arctan2(v[2], v[0])

    # Get random starting position and time of flight to the wall
    x = np.random.uniform(0, d)
    tau_hit = (d * 0.5 *(1 + direction) - x) / v[0]
    Delta_tau = d/np.abs(v[0])                      # Time of flight between walls

    # Get random lifetime
    r_tau = np.random.random()
    tau_0 = -tau_0_0 * np.log(1 - r_tau)

    # Counters

    reflection = 0
    transmitted = 0
    tau_0_ind = int(round((time_intervals-1)*tau_0/tau_max))
    max_tau_ind = min(tau_0_ind,time_intervals)

    for i in range(max_tau_ind):
        if t[i] >= tau_hit:

            T = PT(theta_prime, alpha)
            #print "T:",T
            #print "Theta prime:",theta_prime*(180./np.pi)
            beta_min = -np.pi/2
            beta_max =  np.pi/2
            if T < 0: # We have shadowing and so need to adjust the limits on the
                      # possible incidence angles Beta
                if np.sin(theta_prime) * np.cos(theta_prime) >= 0:
                    beta_min = np.arcsin(1 - (2 * np.abs(np.cos(theta_prime)))/alpha)
                else:
                    beta_max = -np.arcsin(1 - (2 * np.abs(np.cos(theta_prime)))/alpha)
                beta = randomBetaAngle(beta_min, beta_max)
                reflection+=1
            else:
                if np.random.random() < T:  # Transmitted
                    beta = np.pi/2
                    transmitted+=1
                else:                       # Scattered
                    beta = randomBetaAngle(beta_min, beta_max)
                    reflection+=1
            theta_prime += np.pi - (2*beta) # New dirrection of travel in the x-z plance
            v[0] = vxz * np.cos(theta_prime)
            v[2] = vxz * np.sin(theta_prime)
            tau_hit += d/np.abs(v[0])
        J[:,i] = v

    c  = ACF(t, J)
    I   = np.array([integrate.cumtrapz(c[i], tau, initial = 0) for i in range(3)])

    return c, I


def RandomHalfSpaceDirection(normal):
    # Get random direction
    theta, phi = RandomDirection()
    nv = np.array([np.cos(theta), np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi)])
    v  = v_g * nv
    direction = float(np.sign(v[0]))
    vxz = np.sqrt((v[0])**2 + (v[2])**2)
    theta_prime = np.arctan2(v[2], v[0])


def getRandomPhononDiffCyli(tau_max, time_intervals, v_g, tau_0_0, d, alpha):

    J = np.zeros([3,time_intervals])
    t = np.linspace(0, tau_max, time_intervals)

    # Get random direction
    theta, phi = RandomDirection()
    nv = np.array([np.cos(theta), np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi)])
    v  = v_g * nv
    direction = float(np.sign(v[0]))
    vxz = np.sqrt((v[0])**2 + (v[2])**2)
    theta_prime = np.arctan2(v[2], v[0])

    # Get random starting position and time of flight to the wall
    x = np.random.uniform(0, d)
    tau_hit = (d * 0.5 *(1 + direction) - x) / v[0]
    Delta_tau = d/np.abs(v[0])                      # Time of flight between walls

    # Get random lifetime
    r_tau = np.random.random()
    tau_0 = -tau_0_0 * np.log(1 - r_tau)

    # Counters

    reflection = 0
    transmitted = 0
    tau_0_ind = int(round((time_intervals-1)*tau_0/tau_max))
    max_tau_ind = min(tau_0_ind,time_intervals)

    for i in range(max_tau_ind):
        if t[i] >= tau_hit:

            T = PT(theta_prime, alpha)
            #print "T:",T
            #print "Theta prime:",theta_prime*(180./np.pi)
            beta_min = -np.pi/2
            beta_max =  np.pi/2
            if T < 0: # We have shadowing and so need to adjust the limits on the
                      # possible incidence angles Beta
                if np.sin(theta_prime) * np.cos(theta_prime) >= 0:
                    beta_min = np.arcsin(1 - (2 * np.abs(np.cos(theta_prime)))/alpha)
                else:
                    beta_max = -np.arcsin(1 - (2 * np.abs(np.cos(theta_prime)))/alpha)
                beta = randomBetaAngle(beta_min, beta_max)
                reflection+=1
            else:
                if np.random.random() < T:  # Transmitted
                    beta = np.pi/2
                    transmitted+=1
                else:                       # Scattered
                    beta = randomBetaAngle(beta_min, beta_max)
                    reflection+=1
            theta_prime += np.pi - (2*beta) # New dirrection of travel in the x-z plance
            v[0] = vxz * np.cos(theta_prime)
            v[2] = vxz * np.sin(theta_prime)
            tau_hit += d/np.abs(v[0])
        J[:,i] = v

    c  = ACF(t, J)
    I   = np.array([integrate.cumtrapz(c[i], tau, initial = 0) for i in range(3)])

    return c, I


def averagePhonons(nblocks, samplesPerBlock, t_max, time_intervals, tau_mean, v_g, d, alpha):
    # Average over nblock of samplesPerBlock
    ACF   = np.zeros([3, nblocks, time_intervals])
    kappa = np.zeros([3, nblocks, time_intervals])
    ACF_fft   = np.zeros([3, nblocks, time_intervals])
    kappa_fft = np.zeros([3, nblocks, int(1+(time_intervals-1)/2)])

    for i in range(nblocks):
        for j in range(samplesPerBlock):
            c, I = getRandomPhononSpecWall(t_max, time_intervals, v_g, tau_mean, d, alpha)

            ACF[:,i]   += c
            kappa[:,i] += I

    ACF   /= tau_mean*float(samplesPerBlock)
    kappa /= tau_mean*float(samplesPerBlock)

    ACFm = ACF.mean(axis=1)
    ACFe = ACF.std(axis=1)/np.sqrt(float(nblocks-1))
    km   = kappa.mean(axis=1)
    ke   = kappa.std(axis=1)/np.sqrt(float(nblocks-1))
    ACFm_fft = ACF_fft.mean(axis=1)
    ACFe_fft = ACF_fft.std(axis=1)/np.sqrt(float(nblocks-1))
    km_fft   = kappa_fft.mean(axis=1)
    ke_fft   = kappa_fft.std(axis=1)/np.sqrt(float(nblocks-1))
    return ACFm, ACFe, km, ke, ACFm_fft, ACFe_fft, km_fft, ke_fft
