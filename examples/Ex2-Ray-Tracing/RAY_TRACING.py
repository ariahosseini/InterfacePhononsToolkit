"""

A Example of running simulation to model anticorrelation effect in materials containing nanoscale porosity

"""

# Import ray_trying method, the required libraries are automatically imported
from util.generate_figs import *
from util.ray_tracing import *
import time


# Define a function to write a block of data as a big file
def WriteDataFile(outfile, v_g, tau_mean, Lp, tau_max, alphas, data, uncertainty, time_intervals):
    """
    Generate the output file

    :arg
        outfile                                 : Output file's name
        time_intervals                          : Time intervals
        v_g                                     : Phonon group velocity (magnitude)
        alphas                                  :
        tau_max                                 : Maximum correlation time to compute the ACF out to
        tau_mean                                : Phonon's average lifetime
        Lp                                      : Spacing
        data                                    :
        uncertainty                             :

    :returns
        This is to dump the output file, no return Obj.
    """

    paramkey = 'Parameters: v_g='+str(v_g)+' tau_mean='+str(tau_mean)+' Lp='+str(Lp)+' tau_max='+str(tau_max) +\
               ' Kn='+str(v_g*tau_mean/Lp)+'\n'

    headkey = ("Time", "data[0]", "data[1]", "data[...]", "uncertainty[0]", "uncertainty[1]", "uncertainty[...]")
    separator = ' '
    with open(outfile, 'w') as writer:
        writer.write(paramkey)
        writer.write('alphas:'+separator+separator.join(map(str, alphas))+"\n")
        writer.write(separator.join(headkey)+"\n")
        for i in range(time_intervals):
            time = tau_max * i/float(time_intervals)
            writer.write(str(time)+separator+separator.join(map(str, data[:,i])) +
                         separator+separator.join(map(str, uncertainty[:, i]))+"\n")


# Define initial parameters
tau_mean = 0.5                          # Mean lifetime
v_g = 5.0                               # Group velocity
Lp = 200                                # Spacing between pores
k_bulk = tau_mean * (v_g**2) / 3.0      # Bulk thermal conductivity

# Simulation parameters
tau_max = 25 * tau_mean                 # Maximum correlation time to compute the ACF out to
time_intervals = 10001                  # Time intervals
nblocks = 4                             # Number of blocks to average
samplesPerBlock = 500                   # Number of samples in each block


# Define sweep parameter
alphas = np.linspace(0.0, 1, 2)

# Correlation time
tau = np.linspace(0.0, tau_max, time_intervals)
Jn = 7
J = np.asarray([[np.sign(tau[i] - Jn * tau_mean) for i in range(time_intervals)] for j in range(3)])      # Flux

tic = time.time()
acf = ACF(tau, J)                       # Compute autocorrelation
toc = time.time()

print('timing = ' + str(toc - tic))
print(np.shape(J), np.shape(tau))

# Plot heat flux components along Cartesian axes
plt.figure()
plt.plot(tau, J[0, :])
plt.plot(tau, J[1, :])
plt.plot(tau, J[2, :])
plt.show()

# Plot autocorrelation components along Cartesian axes
quickmultiplot([[tau, acf[i]] for i in range(3)])

tic = time.time()
acf_fft = ACF_FFT(tau, J)               # Compute autocorrelation using discrete fourier transform
toc = time.time()

print('timing = ' + str(toc - tic))

# Plot autocorrelation components along Cartesian axes
quickmultiplot([[tau, acf_fft[i]] for i in range(3)])

# Compare autocorrelation components along x using the two previous methods
quickmultiplot([[tau, acf_fft[0]], [tau, acf[0]]])

tic = time.time()
# Compute autocorrelation function and thermal conductivity for phonons flying between walls with spectral scattering
acf, I = getRandomPhononSpecWall(tau_max, time_intervals, v_g, tau_mean, Lp, 0)

# Compute average and standard deviation of autocorrelation and thermal conductivity
ACFm, ACFe, km, ke = averagePhonons(nblocks, samplesPerBlock, tau_max, time_intervals,
                                    v_g, tau_mean, Lp, 0, mode="SpecWall")
toc = time.time()

print('timing = '+str(toc - tic))
print('kappa_bulk = '+str(k_bulk))
print('kappa_xyz = '+str(km[:, -1]))
print('timing = '+str(toc - tic))
print(tau_mean*v_g**2/3.0)
print(km[:, -1])

# Plot correlation along x
quickmultiplot([[tau, ACFm[0]]])
# Plot mean and standard deviation of correlation along Cartesian axes
quickmultierrplot([[tau, ACFm[i], ACFe[i]] for i in range(3)])
# Plot mean and standard deviation of conductivity along Cartesian axes
quickmultierrplot([[tau, km[i], ke[i]] for i in range(3)])


# START SIMULATION: Perform a sweeping over alpha

# Define correlation time
tau = np.linspace(0.0,tau_max,time_intervals)

# Allocate memory to correlation and thermal conductivity blocks
mACFtab = np.zeros([len(alphas), 3, time_intervals])
eACFtab = np.zeros([len(alphas), 3, time_intervals])
mktab = np.zeros([len(alphas), 3, time_intervals])
ektab = np.zeros([len(alphas), 3, time_intervals])


for alphindex in range(len(alphas)):
    alpha = alphas[alphindex]
    print('alpha: '+str(alpha))          # Print sweep parameter

    ACFm, ACFe, km, ke = averagePhonons(nblocks, samplesPerBlock, tau_max,
                                        time_intervals, tau_mean, v_g, Lp,
                                        alpha, mode="SpecWall")        # Compute correlation and thermal conductivity

    quickmultierrplot([[tau, ACFm[i], ACFe[i]] for i in range(3)])     # Plot correlation along axes
    quickmultierrplot([[tau, km[i], ke[i]] for i in range(3)])         # Plot conductivity along axes
    print('kappa_bulk = '+str(tau_mean*v_g**2/3.0))
    print('kappa_xyz = '+str(km[:, -1]))

    mACFtab[alphindex] = ACFm                                          # Mean correlation
    eACFtab[alphindex] = ACFe                                          # Standard deviation of correlation
    mktab[alphindex] = km                                              # Mean thermal conductivity
    ektab[alphindex] = ke                                              # Standard deviation of thermal conductivity

print('kappa_bulk = '+str(tau_mean*v_g**2/3.0))
print('kappa_xyz(alpha='+str(alphas[0])+') = '+str(mktab[0, :, -1]))
print('kappa_xyz(alpha='+str(alphas[-1])+') = '+str(mktab[-1, :, -1]))
print('kappa_xyz = '+str(mktab[0, :, -1]))


# Save the data as a big file

ReflectionTag = 'specular-wall'
KNtag = str(v_g*tau_mean/Lp)
dirtag = ('x', 'y', 'z')

# Generate output file
for i in range(3):
    outfile = ReflectionTag+'-ACF-'+dirtag[i]+'-Kn'+KNtag+'.txt'
    WriteDataFile(outfile, v_g, tau_mean, Lp, tau_max, alphas, mACFtab[:, i], eACFtab[:, i], time_intervals)
    outfile = ReflectionTag+'-kappa-'+dirtag[i]+'-Kn'+KNtag+'.txt'
