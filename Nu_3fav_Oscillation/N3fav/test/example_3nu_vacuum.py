

from __future__ import print_function


import sys
sys.path.append('../src')

import numpy as np

import oscprob3nu
import hamiltonians3nu
from globaldefs import *

energy = 1.e9     # Neutrino energy [eV]
baseline = 1.3e3  # Baseline [km]

# Use the NuFit 4.0 best-fit values of the mixing parameters pulled from
# globaldefs.  NO means "normal ordering"; change NO to IO if you want
# to use inverted ordering.
h_vacuum_energy_indep = \
    hamiltonians3nu.hamiltonian_3nu_vacuum_energy_independent(  S12_NO_BF,
                                                                S23_NO_BF,
                                                                S13_NO_BF,
                                                                DCP_NO_BF,
                                                                D21_NO_BF,
                                                                D31_NO_BF)
h_vacuum = np.multiply(1./energy, h_vacuum_energy_indep)

# CONV_KM_TO_INV_EV is pulled from globaldefs; it converts km to eV^{-1}
Pee, Pem, Pet, Pme, Pmm, Pmt, Pte, Ptm, Ptt = \
    oscprob3nu.probabilities_3nu( h_vacuum, baseline*CONV_KM_TO_INV_EV)

print("Pee = %6.5f, Pem = %6.5f, Pet = %6.5f" % (Pee, Pem, Pet))
print("Pme = %6.5f, Pmm = %6.5f, Pmt = %6.5f" % (Pme, Pmm, Pmt))
print("Pte = %6.5f, Ptm = %6.5f, Ptt = %6.5f" % (Pte, Ptm, Ptt))
