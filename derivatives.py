import numpy as np
import thermodynamics as thermo
from constants import me, mpl
import weakrates as weak
import expansion as ex
import nseabundance as nse
######################
#
# to be used in conjuction with ODESolve.py
#
# dy/dx = f(x, y, p)
#
# p is either a number or an array of constant parameters used in the derivatives
# x is the independent variable
# y is an array of the dependent variables
#
#####################


#####################
# Example:
# dA/dx = B
# dB/dt = C
# dC/dt = D
# dD/dt = pi^4 A
#####################

#Hubble expansion rate

# outputs first 3 elements of array and then an array of the last 9
def sep(z):
    return z[0], z[1], z[2], z[3:]
T,t,eta,A = sep(np.arange(12))

#creates array of 12 dependent variables
def depvar(T,t,eta,A):
    return np.concatenate((np.array([T,t,eta]), A))
    

def f(a,y,p):
    T, t, eta, A = sep(y)
    der=np.zeros(3)
    d=np.zeros(9)

    der[0] = -(3*a**2*ex.sth(T))/(a**3*ex.dsthdT(T))
    der[1] = (1/a)*((8*np.pi*ex.ptot(T,a))/(3*(mpl)**2))**(-1/2)
    der[2] = -3*eta*((1/T)*der[0]+(1/a))

    d[nse.P_INDEX] += A[nse.N_INDEX]*weak.Nnptot(T,a)
    d[nse.P_INDEX] -= A[nse.P_INDEX]*weak.Npntot(T,a)
    d[nse.N_INDEX] -= A[nse.N_INDEX]*weak.Nnptot(T,a)
    d[nse.N_INDEX] += A[nse.P_INDEX]*weak.Npntot(T,a)
    d[nse.H2_INDEX] = 0
    d[nse.H3_INDEX] = 0
    d[nse.HE3_INDEX] = 0
    d[nse.HE4_INDEX] = 0
    d[nse.BE7_INDEX] = 0
    d[nse.LI7_INDEX] = 0
    d[nse.LI6_INDEX] = 0

    return depvar(der[0], der[1], der[2], d*der[1])