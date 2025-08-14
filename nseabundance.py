#NSE abundance
import numba as nb
import numpy as np
from constants import zeta3, mN


# multiplicity, binding energy, atomic number, mass number
g = np.array([2, 2 ,3, 2, 2, 1, 4, 4, 3], dtype=np.double)
B = np.array([0, 0, 2.22, 8.48, 7.72, 28.3, 37.6, 39.25, 31.99], dtype=np.double)
Z = np.array([1, 0, 1, 1, 2, 2, 4, 3, 3], dtype=np.double)
A = np.array([1, 1, 2, 3, 3, 4, 7, 7, 6], dtype=np.double)

P_INDEX = 0
N_INDEX = 1
H2_INDEX = 2
H3_INDEX = 3
HE3_INDEX = 4
HE4_INDEX = 5
BE7_INDEX = 6
LI7_INDEX = 7
LI6_INDEX = 8

PNGD_INDEX = 0
TPGA_INDEX = 1

@nb.njit()
# NSE function
def nse(T, eta, Yp, Yn):
  
    nse=np.zeros(9)
    
    for i in range(len(nse)):
        nB = eta*(3/2)*zeta3*T**3
        epsilon=(1/2)*((2*np.pi)/(mN*T))**(3/2)*nB
        nse[i]=(1/2)*Yp**Z[i]*Yn**(A[i]-Z[i])*g[i]*A[i]**(3/2)*epsilon**(A[i]-1)*np.exp(B[i]/T)
        
    return nse
