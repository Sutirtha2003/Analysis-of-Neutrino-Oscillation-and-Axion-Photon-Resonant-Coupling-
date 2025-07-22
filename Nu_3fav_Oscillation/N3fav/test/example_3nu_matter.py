

from __future__ import print_function



import sys
sys.path.append('../src')

import numpy as np

import oscprob3nu
import hamiltonians3nu
from globaldefs import *

energy = 1.e9     # Neutrino energy [eV]
baseline = 1.3e3  # Baseline [km]

h_vacuum_energy_indep = \
    hamiltonians3nu.hamiltonian_3nu_vacuum_energy_independent(  S12_NO_BF,
                                                                S23_NO_BF,
                                                                S13_NO_BF,
                                                                DCP_NO_BF,
                                                                D21_NO_BF,
                                                                D31_NO_BF)

# Units of VCC_EARTH_CRUST: [eV]
h_matter = hamiltonians3nu.hamiltonian_3nu_matter(  h_vacuum_energy_indep,
                                                    energy,
                                                    VCC_EARTH_CRUST)

Pee, Pem, Pet, Pme, Pmm, Pmt, Pte, Ptm, Ptt = \
    oscprob3nu.probabilities_3nu(h_matter, baseline*CONV_KM_TO_INV_EV)

print("Pee = %6.5f, Pem = %6.5f, Pet = %6.5f" % (Pee, Pem, Pet))
print("Pme = %6.5f, Pmm = %6.5f, Pmt = %6.5f" % (Pme, Pmm, Pmt))
print("Pte = %6.5f, Ptm = %6.5f, Ptt = %6.5f" % (Pte, Ptm, Ptt))
