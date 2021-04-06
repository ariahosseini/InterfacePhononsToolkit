# USER GUIDE

<div align="justify">
  
<p>
  
Following examples show how to use *InterfacePhononsToolkit* librarier to model (Ex1)  wavepacket (Ex2) anticorrelation effect in materials containing nanoscale porosity.
</p>

## Ex1-Phonon-Wavepacket
  
  ### Roadmap
<p>  
  [1] Run in.Si-hessian, note that in.hessian, SiCGe.tersoff, data.Si, dump.py and forcedump2hessian.py files should be in the directory where lammps runs.
  
  [2] Previous step generates Si-hessian-mass-weighted-hessian.d and data.Si-unwrapped files. These two are the input file for WAVEPACKET.py. Run WAVEPACKET.py.
  
  [3] Dump the wavepacket data file and use it add the initial atom position in Lammps. The lammps input file is in.Si.
  
  [4] Previous step generates pad.d which include atoms kinetic energy during running time.
  
  Here I plotted the mean kinetic energy of atoms in layes along z
  
  <p align="center">
<img src="../figs/KE-08.png" align="center" alt="drawing" width="700px"> 
</p>

If the python code  is run for different qpoints, it can be used to compute the spectral transmission coefficient.

<p align="center">
<img src="../figs/Transmission_si_ge.png" align="center" alt="drawing" width="700px"> 
</p>


</p>

## Ex2-Ray-Tracing
  
  ### Roadmap
  Simply run RAY_TRACING.py, a load of plots will be generated and output files containing thermal conductivity and anticorrelations will be dumped. For more information see 
  
  - de Sousa Oliveira, L., Hosseini, S. A., Greaney, A., \& Neophytou, N. (2020). Heat current anticorrelation effects leading to thermal conductivity reduction in nanoporous Si. Physical Review B, 102(20), 205405. https://journals.aps.org/prb/abstract/10.1103/PhysRevB.102.205405
