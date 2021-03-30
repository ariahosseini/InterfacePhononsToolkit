def RandomDirection():
    theta = np.arccos(1 - (2*np.random.uniform(0, 1)))
    phi = np.random.uniform(-np.pi, np.pi)
    return theta, phi

TimeIndex = lambda time, maxtime, timesteps: int(round((timesteps-1)*time/maxtime))

def getRandomPhononSpecWall(tau_max, time_intervals, v_g, tau_0_0, d, P_reflection):

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
    hit_index = TimeIndex(tau_hit, tau_max, time_intervals)

    Delta_tau = d/np.abs(v[0])                      # Time of flight between walls

    # Get random lifetime
    r_tau = np.random.random()
    tau_0 = -tau_0_0 * np.log(1 - r_tau)
    #print(tau_0, tau_max, time_intervals)
    die_index = min(TimeIndex(tau_0, tau_max, time_intervals),time_intervals)
    #print('die_index='+str(die_index))
    now_index = 0

    # counters
    hit_count   = 0
    #transmitted = 0
    #reflected   = 0

    #print('tau_0: '+str(tau_0)+' tau_0_0: '+str(tau_0_0)+' tau_max: '+str(tau_max)+' time_intervals: '+str(time_intervals)+'. v = '+str(v))
    #print('tau_hit: '+str(tau_hit)+' Delta_tau: '+str(Delta_tau))
    #print('hit count: '+str(hit_count)+'. now, hit, and die indices: '+str(now_index)+', '+str(hit_index)+', '+str(die_index))

    while hit_index < die_index:

        #J[:,now_index:hit_index] = np.transpose(np.matmul(np.ones([hit_index-now_index,1]),[v]))
        row = np.ones(hit_index-now_index)
        J[:,now_index:hit_index] = np.array([v[i]*row for i in range(3)])
        hit_count += 1

        dice = np.random.random()
        if dice < P_reflection:
        #    reflected += 1
            direction *= -1.0
            v[0]      *= -1.0
        #else:
        #    transmitted += 1

        tau_hit  += Delta_tau
        now_index = hit_index

        hit_index = TimeIndex(tau_hit, tau_max, time_intervals)
        #print('hit count: '+str(hit_count)+'. now, hit, and die indice: '+str(now_index)+', '+str(hit_index)+', '+str(die_index))

    #J[:,now_index:die_index] = np.transpose(np.matmul(np.ones([die_index-now_index,1]),[v]))
    row = np.ones(die_index-now_index)
    J[:,now_index:die_index] = np.array([v[i]*row for i in range(3)])
    #print('hit count: '+str(hit_count)+'. now, hit, and die indice: '+str(now_index)+', '+str(hit_index)+', '+str(die_index))
    #print('reflected: '+str(reflected)+' transmitted: '+str(transmitted)+' tau_max: '+str(tau_max)+'. v = '+str(v))
    #quickmultiplot([[t,J[i]] for i in range(3)])
    c  = ACF(t, J)
    #cfft  = ACF_FFT(t, J)
#    quickmultiplot([[t[:1000],c[i,:1000]] for i in range(3)])
#    quickmultiplot([[t[:1000],cfft[i,:1000]] for i in range(3)])
#    quickmultiplot([[t[:1000],c[0,:1000]],[t[:1000],cfft[0,:1000]]])
    I   = np.array([integrate.cumtrapz(c[i], tau, initial = 0) for i in range(3)])
    #t2_ind = (1+(time_intervals-1)/2)
    #Ifft   = np.array([integrate.cumtrapz(c[i,:t2_ind], t[:t2_ind], initial = 0) for i in range(3)])
#    print(I[:,-1])
#    print(Ifft[:,-1])
#    print(c[0,0]/cfft[0,0])
#    print(I[0,-1]/Ifft[0,-1])
#    return c, I, cfft, Ifft
    return c, I

def getRandomPhononSpecWall_fft(tau_max, time_intervals, v_g, tau_0_0, d, P_reflection):

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
    hit_index = TimeIndex(tau_hit, tau_max, time_intervals)

    Delta_tau = d/np.abs(v[0])                      # Time of flight between walls

    # Get random lifetime
    r_tau = np.random.random()
    tau_0 = -tau_0_0 * np.log(1 - r_tau)
    #print(tau_0, tau_max, time_intervals)
    die_index = min(TimeIndex(tau_0, tau_max, time_intervals),time_intervals)
    #print('die_index='+str(die_index))
    now_index = 0

    # counters
    hit_count   = 0
    #transmitted = 0
    #reflected   = 0

    #print('tau_0: '+str(tau_0)+' tau_0_0: '+str(tau_0_0)+' tau_max: '+str(tau_max)+' time_intervals: '+str(time_intervals)+'. v = '+str(v))
    #print('tau_hit: '+str(tau_hit)+' Delta_tau: '+str(Delta_tau))
    #print('hit count: '+str(hit_count)+'. now, hit, and die indices: '+str(now_index)+', '+str(hit_index)+', '+str(die_index))

    while hit_index < die_index:

        #J[:,now_index:hit_index] = np.transpose(np.matmul(np.ones([hit_index-now_index,1]),[v]))
        row = np.ones(hit_index-now_index)
        J[:,now_index:hit_index] = np.array([v[i]*row for i in range(3)])
        hit_count += 1

        dice = np.random.random()
        if dice < P_reflection:
        #    reflected += 1
            direction *= -1.0
            v[0]      *= -1.0
        #else:
        #    transmitted += 1

        tau_hit  += Delta_tau
        now_index = hit_index

        hit_index = TimeIndex(tau_hit, tau_max, time_intervals)
        #print('hit count: '+str(hit_count)+'. now, hit, and die indice: '+str(now_index)+', '+str(hit_index)+', '+str(die_index))

    #J[:,now_index:die_index] = np.transpose(np.matmul(np.ones([die_index-now_index,1]),[v]))
    row = np.ones(die_index-now_index)
    J[:,now_index:die_index] = np.array([v[i]*row for i in range(3)])
    #print('hit count: '+str(hit_count)+'. now, hit, and die indice: '+str(now_index)+', '+str(hit_index)+', '+str(die_index))
    #print('reflected: '+str(reflected)+' transmitted: '+str(transmitted)+' tau_max: '+str(tau_max)+'. v = '+str(v))
    #quickmultiplot([[t,J[i]] for i in range(3)])
    #c  = ACF(t, J)
    cfft  = ACF_FFT(t, J)
#    quickmultiplot([[t[:1000],c[i,:1000]] for i in range(3)])
#    quickmultiplot([[t[:1000],cfft[i,:1000]] for i in range(3)])
#    quickmultiplot([[t[:1000],c[0,:1000]],[t[:1000],cfft[0,:1000]]])
    #I   = np.array([integrate.cumtrapz(c[i], tau, initial = 0) for i in range(3)])
    t2_ind = (1+(time_intervals-1)/2)
    Ifft   = np.array([integrate.cumtrapz(c[i,:t2_ind], t[:t2_ind], initial = 0) for i in range(3)])
#    print(I[:,-1])
#    print(Ifft[:,-1])
#    print(c[0,0]/cfft[0,0])
#    print(I[0,-1]/Ifft[0,-1])
#    return c, I, cfft, Ifft
    return cfft, Ifft

def getRandomPhononSpecWallOld(tau_max, time_intervals, v_g, tau_0_0, d, P_reflection):

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
    Delta_tau = d/np.abs(v[0])                      # Time of flight between walls

    # Get random lifetime
    r_tau = np.random.random()
    tau_0 = -tau_0_0 * np.log(1 - r_tau)

    tau_0_ind = int(round((time_intervals-1)*tau_0/tau_max))
    max_tau_ind = min(tau_0_ind,time_intervals)
    for i in range(max_tau_ind):
        if t[i] >= tau_hit:
            tau_hit += Delta_tau
            dice = np.random.random()
            if dice < P_reflection:
                direction *= -1.0
                v[0] *= -1.0
        J[:,i] = v

    c  = ACF(t, J)
    I   = np.array([integrate.cumtrapz(c[i], tau, initial = 0) for i in range(3)])
    return c, I

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

#def getRandomPhononCylinder(t_max, time_intervals, tau_0_0, v_g, d, alpha):
#   # print v_g
#    #Omega_x = np.zeros(time_intervals)
##    J = np.zeros(3,time_intervals)
#    J_x = np.zeros(time_intervals)
#    J_y = np.zeros(time_intervals)
#    J_z = np.zeros(time_intervals)
#
#    t = np.linspace(0, t_max, time_intervals)
#    r_Omega = np.random.uniform(0, 1)
#    theta = np.arccos(1 - (2*r_Omega))
#    phi = np.random.uniform(-np.pi, np.pi)
#    v_x = v_g * np.cos(theta)
#    v_y = v_g * (np.sin(theta) * np.cos(phi))
#    v_z = v_g * (np.sin(theta)*np.sin(phi))
#    v_x_z = np.sqrt((v_x)**2 + (v_z)**2)
#    theta_prime = np.arctan2(v_z, v_x)
#    x = np.random.uniform(0, d)
#   # r = x * np.cos(theta)
#    transmitted = 0
#    reflection = 0
#    tau_prime = (d * 0.5 *(1 + np.sign(v_x)) - x) / v_x
#
#    r_tau = np.random.random()
#    tau_0 = -tau_0_0 * np.log(1 - r_tau)
#
#    max_t_ind = min(1+int((time_intervals-1)*tau_0/float(tau_max)),time_intervals)
#    #print(max_t_ind)
#    #for i in range(time_intervals):
#    for i in range(max_t_ind):
#        if t[i] >= tau_prime:
#            T = PT(theta_prime, alpha)
#            #print "T:",T
#            #print "Theta prime:",theta_prime*(180./np.pi)
#            beta_min = -np.pi/2
#            beta_max =  np.pi/2
#            R = np.random.random()
#            if T < 0:
#                if np.sin(theta_prime) * np.cos(theta_prime) >= 0:
#                    beta_min = np.arcsin(1 - (2 * np.abs(np.cos(theta_prime)))/alpha)
#                else:
#                    beta_max = -np.arcsin(1 - (2 * np.abs(np.cos(theta_prime)))/alpha)
#                beta = rand_betaFunction(R, beta_min, beta_max)
#                reflection+=1
#            else:
#                if np.random.random() < T:
#                    beta = np.pi/2
#                    transmitted+=1
#                else:
#                    beta = rand_betaFunction(R, beta_min, beta_max)
#                    reflection+=1
#            theta_prime += np.pi - (2*beta)
#            v_x = v_x_z * np.cos(theta_prime)
#            v_z = v_x_z * np.sin(theta_prime)
#            tau_prime += d/np.abs(v_x)




def averagePhonons(nBlocks, samplesPerBlock, t_max, time_intervals, tau_0_0, v_g, d, alpha):
    # Average over nBlock of samplesPerBlock
    ACF   = np.zeros([3, nBlocks, time_intervals])
    kappa = np.zeros([3, nBlocks, time_intervals])
    ACF_fft   = np.zeros([3, nBlocks, time_intervals])
    kappa_fft = np.zeros([3, nBlocks, 1+(time_intervals-1)/2])

    for i in range(nBlocks):
        #print(i)
        for j in range(samplesPerBlock):
            #print((i,j))
#            c, I, cfft, Ifft = getRandomPhononSpecWall(t_max, time_intervals, v_g, tau_0_0, d, alpha)
            c, I = getRandomPhononSpecWall(t_max, time_intervals, v_g, tau_0_0, d, alpha)
            #ccfft, Ifft = getRandomPhononSpecWall_fft(t_max, time_intervals, v_g, tau_0_0, d, alpha)
#            c, I = getRandomPhononSpecWallOld(t_max, time_intervals, v_g, tau_0_0, d, alpha)
#            c, I = getRandomPhononDiffWall(t_max, time_intervals, v_g, tau_0_0, d, alpha)
#            c, I = getRandomPhononSpecCyli(t_max, time_intervals, v_g, tau_0_0, d, alpha)
            ACF[:,i]   += c
            kappa[:,i] += I
            #ACF_fft[:,i]   += cfft
            #kappa_fft[:,i] += Ifft
    ACF   /= tau_0_0*float(samplesPerBlock)
    kappa /= tau_0_0*float(samplesPerBlock)
    #ACF_fft   /= tau_0_0*float(samplesPerBlock)
    #kappa_fft /= tau_0_0*float(samplesPerBlock)

    ACFm = ACF.mean(axis=1)
    ACFe = ACF.std(axis=1)/np.sqrt(float(nblocks-1))
    km   = kappa.mean(axis=1)
    ke   = kappa.std(axis=1)/np.sqrt(float(nblocks-1))
    #ACFm_fft = ACF_fft.mean(axis=1)
    #ACFe_fft = ACF_fft.std(axis=1)/np.sqrt(float(nblocks-1))
    #km_fft   = kappa_fft.mean(axis=1)
    #ke_fft   = kappa_fft.std(axis=1)/np.sqrt(float(nblocks-1))
    return ACFm, ACFe, km, ke, ACFm_fft, ACFe_fft, km_fft, ke_fft
