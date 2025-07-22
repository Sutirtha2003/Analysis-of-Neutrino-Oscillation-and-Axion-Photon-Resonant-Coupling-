

from __future__ import print_function

__version__ = "1.0"
__author__ = "Mauricio Bustamante"
__email__ = "mbustamante@gmail.com"


import sys
sys.path.append('../src')

import oscprob2nu


hamiltonian = [
                [1.0+0.0j, 1.0+2.0j],
                [1.0-2.0j, 3.0+0.0j]
]

L = 1.0

Pee, Pem, Pme, Pmm = oscprob2nu.probabilities_2nu(hamiltonian, L)

print("Pee = %6.5f, Pem = %6.5f" % (Pee, Pem))
print("Pme = %6.5f, Pmm = %6.5f" % (Pme, Pmm))
