import numba as nb
import numpy as np
import thermodynamics as thermo
from constants import me, mpl
import weakrates as weak

@nb.njit()
def sth(T):
    sth = ((2*np.pi**2*2*T**3)/45)+thermo.s(T,me,2)+thermo.s(T,me,2)
    return sth

@nb.njit()
def dsthdT(T):
    dsthdT = (6*np.pi**2*2*T**2)/45+thermo.dsdT(T,me,2)+thermo.dsdT(T,me,2)
    return dsthdT

@nb.njit()
def ptot(T,a):
    ptot = (np.pi**2*2*T**4)/30+thermo.p(T,me,2)+thermo.p(T,me,2)+thermo.p(1/a,0,6)
    return ptot

@nb.njit()
def H(T,a):
    mpl=1.22e22
    H=(((8*np.pi)/(3*mpl**2))*ptot(T,a))**(1/2)
    return H