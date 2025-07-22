

from __future__ import print_function



import sys
sys.path.append('../src')

import oscprob3nu

hamiltonian = [
                [1.0+0.0j, 0.0+2.0j, 0.0-1.0j],
                [0.0-2.0j, 3.0+0.0j, 3.0+0.0j],
                [0.0+1.0j, 3.0-0.0j, 5.0+0.0j]
]

L = 1.0

Pee, Pem, Pet, Pme, Pmm, Pmt, Pte, Ptm, Ptt = \
    oscprob3nu.probabilities_3nu(hamiltonian, L)

print("Pee = %6.5f, Pem = %6.5f, Pet = %6.5f" % (Pee, Pem, Pet))
print("Pme = %6.5f, Pmm = %6.5f, Pmt = %6.5f" % (Pme, Pmm, Pmt))
print("Pte = %6.5f, Ptm = %6.5f, Ptt = %6.5f" % (Pte, Ptm, Ptt))
