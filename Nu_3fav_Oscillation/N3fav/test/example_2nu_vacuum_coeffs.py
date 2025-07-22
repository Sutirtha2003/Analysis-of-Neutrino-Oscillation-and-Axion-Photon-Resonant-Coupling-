
import sys
sys.path.append('../src')

import numpy as np

import oscprob2nu
import hamiltonians2nu
from globaldefs import *

energy = 1.e9     # Neutrino energy [eV]
baseline = 1.3e3  # Baseline [km]

h_vacuum_energy_indep = \
    hamiltonians2nu.hamiltonian_2nu_vacuum_energy_independent(  S23_NO_BF,
                                                                D31_NO_BF)
h_vacuum = np.multiply(1./energy, h_vacuum_energy_indep)

h1, h2, h3 = oscprob2nu.hamiltonian_2nu_coefficients(h_vacuum)
print('h1: {:.4e}'.format(h1))
print('h2: {:.4e}'.format(h2))
print('h3: {:.4e}'.format(h3))
print()

evol_operator = oscprob2nu.evolution_operator_2nu(  h_vacuum,
                                                    baseline*CONV_KM_TO_INV_EV)
print('U2 = ')
with np.printoptions(precision=3, suppress=True):
    print(np.array(evol_operator))
