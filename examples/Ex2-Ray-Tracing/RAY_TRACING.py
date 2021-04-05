import numpy as np
import time
import os
# get current directory
from util.autocorrelation_func import *
from util.generate_figs import *
from util.ray_tracing import *

# path = os.getcwd()
# print("Current Directory", path)
# prints parent directory
# print(os.path.abspath(os.path.join(path, os.pardir)))



# Function to write a block of data as a big file
def WriteDataFile(outfile, v_g, tau_mean, Lp, tau_max, alphas, data, uncertainty, time_intervals):

    paramkey = 'Parameters: v_g='+str(v_g)+' tau_mean='+str(tau_mean)+' Lp='+str(Lp)+' tau_max='+str(tau_max)+' Kn='+str(v_g*tau_mean/Lp)+'\n'
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


# Model parameters

tau_mean = 0.5              # Mean lifetime
v_g = 5.0                   # Group velocity
Lp = 200                    # Spacing between pores

k_bulk = tau_mean * (v_g**2) / 3.0      # Bulk thermal conductivity

# Simulation parameters

tau_max = 25 * tau_mean     # Maximum correlation time to compute the ACF out to
time_intervals = 10001      # Time intervals
nblocks = 4
samplesPerBlock = 500


# sweep parameters
alphas = np.linspace(0.0, 1, 21)

tau = np.linspace(0.0, tau_max, time_intervals)     # Lifetime
J = np.asarray([[np.sign(tau[i] - 7 * tau_mean) for i in range(time_intervals)] for j in range(3)])
tic = time.time()
print(np.shape(J), np.shape(tau))
plt.figure()
plt.plot(tau, J[0, :])
plt.plot(tau, J[1, :])
plt.plot(tau, J[2, :])
# plt.show()
#
# exit()
c = ACF(tau, J)
toc = time.time()
print('timing = ' + str(toc - tic))
quickmultiplot([[tau, c[i]] for i in range(3)])
tic = time.time()
cfft = ACF_FFT(tau, J)
print(np.shape(cfft))
# exit()
toc = time.time()
print('timing = ' + str(toc - tic))
quickmultiplot([[tau, cfft[i]] for i in range(3)])
quickmultiplot([[tau, cfft[0]], [tau, c[0]]])
c, I = getRandomPhononSpecWall(tau_max, time_intervals, v_g, tau_mean, Lp, 0)

ACFm, ACFe, km, ke, ACFm_fft, ACFe_fft, km_fft, ke_fft = averagePhonons(nblocks, samplesPerBlock, tau_max, time_intervals, v_g, tau_mean, Lp, 0)
toc = time.time()
print('timing = '+str(toc - tic))

quickmultiplot([[tau,ACFm[0]],[tau,ACFm_fft[0]]])

quickmultierrplot([[tau,ACFm[i],ACFe[i]] for i in range(3)])
quickmultierrplot([[tau,ACFm_fft[i],ACFe_fft[i]] for i in range(3)])
quickmultierrplot([[tau,km[i],ke[i]] for i in range(3)])
quickmultierrplot([[tau,km_fft[i],ke_fft[i]] for i in range(3)])
print('kappa_bulk = '+str(k_bulk))
print('kappa_xyz = '+str(km[:,-1]))
print('kappa_xyz_fft = '+str(km_fft[:,-1]))

print('timing = '+str(toc - tic))

print((tau_mean)*(v_g**2)/3.0)
print(km[:,-1])


# START SIMULATION: Perform a sweeping over alpha

#import timeit
#tic = timeit.timeit()
tau = np.linspace(0.0,tau_max,time_intervals)

mACFtab = np.zeros([len(alphas), 3, time_intervals])
eACFtab = np.zeros([len(alphas), 3, time_intervals])
mktab   = np.zeros([len(alphas), 3, time_intervals])
ektab   = np.zeros([len(alphas), 3, time_intervals])

for alphindex in range(len(alphas)):
   alpha = alphas[alphindex]
   print('alpha: '+str(alpha))

   ACFm, ACFe, km, ke = averagePhonons(nblocks, samplesPerBlock, tau_max, time_intervals, tau_0_0, v_g, d, alpha)
   quickmultierrplot([[tau,ACFm[i],ACFe[i]] for i in range(3)])
   quickmultierrplot([[tau,km[i],ke[i]] for i in range(3)])
   print('kappa_bulk = '+str((tau_0_0)*(v_g**2)/3.0))
   print('kappa_xyz = '+str(km[:,-1]))

   mACFtab[alphindex] = ACFm;
   eACFtab[alphindex] = ACFe;
   mktab[alphindex]   = km;
   ektab[alphindex]   = ke;

#toc = timeit.timeit()
#print('timeing = '+str(tic - toc))
print('kappa_bulk = '+str((tau_0_0)*(v_g**2)/3.0))
print('kappa_xyz(alpha='+str(alphas[0])+') = '+str(mktab[0,:,-1]))
print('kappa_xyz(alpha='+str(alphas[-1])+') = '+str(mktab[-1,:,-1]))
print('kappa_xyz = '+str(mktab[0,:,-1]))


# Save the data as a big file

ReflectionTag = 'specular-wall'
#ReflectionTag = 'diffuse-wall'
#ReflectionTag = 'specular-cylinder'
KNtag  = str(v_g*tau_0_0/d)
dirtag = ('x','y','z')
for i in range(3):
   outfile = ReflectionTag+'-ACF-'+dirtag[i]+'-Kn'+KNtag+'.txt'
   WriteDataFile(outfile, v_g, tau_0_0, d, tau_max, alphas, mACFtab[:,i], eACFtab[:,i])
   outfile = ReflectionTag+'-kappa-'+dirtag[i]+'-Kn'+KNtag+'.txt'
#    WriteDataFile(outfile, v_g, tau_0_0, d, tau_max, alphas, mktab[:,i], ektab[:,i])
