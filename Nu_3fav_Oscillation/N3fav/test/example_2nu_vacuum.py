

from __future__ import print_function




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

Pee, Pem, Pme, Pmm = oscprob2nu.probabilities_2nu(  h_vacuum,
                                                    baseline*CONV_KM_TO_INV_EV)

print("Pee = %6.5f, Pem = %6.5f" % (Pee, Pem))
print("Pme = %6.5f, Pmm = %6.5f" % (Pme, Pmm))
