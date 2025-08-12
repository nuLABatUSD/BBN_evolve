import numpy as np
import thermodynamics as thermo
from constants import me, mpl, zeta3, MeVtoT9, cmgstoMeV, mN
import weakrates as weak
import expansion as ex
import nseabundance as nse
from BBN_linearize import linearize
import xgamma_reactions as xg
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

def sep(z):
    return z[0], z[1], z[2], z[3:]
T,t,eta,A = sep(np.arange(12))

#creates array of 12 dependent variables
def depvar(T,t,eta,A):
    return np.concatenate((np.array([T,t,eta]), A))


def f(a,y,p):
    T, t, eta, A = sep(y)
    nB = eta*(3/2)*zeta3*T**3
   
    Gamma_f=np.zeros(xg.Nrxn)
    fwd1 = np.zeros(xg.Nrnx, dtype=int)
    fwd2 = np.zeros(xg.Nrxn, dtype=int)
    rev = np.zeros(xg.Nrxn, dtype=int)
    Gamma_r=np.zeros(xg.Nrxn)
    
    Gamma_f[nse.PNGD_INDEX] = pngd(T)*nB
    Gamma_f[nse.TPGA_INDEX] = tpga(T)*nB
    fwd1[nse.PNGD_INDEX] = nse.P_INDEX
    fwd1[nse.TPGA_INDEX] = nse.H2_INDEX
    fwd2[nse.PNGD_INDEX] = nse.N_INDEX
    fwd2[nse.TPGA_INDEX] = nse.P_INDEX
    rev[nse.PNGD_INDEX] = nse.H2_INDEX
    rev[nse.TPGA_INDEX] = nse.HE4_INDEX

    
    A_NSE = nse.nse(T, eta, A[nse.P_INDEX], A[nse.N_INDEX])
    d=np.zeros(9)
    
    der=np.zeros(3)

    der[0] = -(3*a**2*ex.sth(T))/(a**3*ex.dsthdT(T))
    der[1] = (1/a)*((8*np.pi*ex.ptot(T,a))/(3*(mpl)**2))**(-1/2)
    der[2] = -3*eta*((1/T)*der[0]+(1/a))
    
    d[nse.P_INDEX] += A[nse.N_INDEX]*weak.Nnptot(T,a)
    d[nse.P_INDEX] -= A[nse.P_INDEX]*weak.Npntot(T,a)
    d[nse.N_INDEX] -= A[nse.N_INDEX]*weak.Nnptot(T,a)
    d[nse.N_INDEX] += A[nse.P_INDEX]*weak.Npntot(T,a)
    d[nse.H3_INDEX] = 0
    d[nse.HE3_INDEX] = 0
    d[nse.HE4_INDEX] = 0
    d[nse.BE7_INDEX] = 0
    d[nse.LI7_INDEX] = 0
    d[nse.LI6_INDEX] = 0
    
    if (p==1):
        d[nse.P_INDEX] -= A[nse.P_INDEX]*A[nse.N_INDEX]*Gamma_f[nse.PNGD_INDEX]
        d[nse.P_INDEX] += A[nse.H2_INDEX]*Gamma_r[nse.PNGD_INDEX]
        d[nse.N_INDEX] -= A[nse.P_INDEX]*A[nse.N_INDEX]*Gamma_f[nse.PNGD_INDEX]
        d[nse.N_INDEX] += A[nse.H2_INDEX]*Gamma_r[nse.PNGD_INDEX]
        d[nse.H2_INDEX] += A[nse.P_INDEX]*A[nse.N_INDEX]*Gamma_f[nse.PNGD_INDEX]
        d[nse.H2_INDEX] -= A[nse.H2_INDEX]*Gamma_r[nse.PNGD_INDEX]
        d[nse.H3_INDEX] += A[nse.H3_INDEX]*A[nse.P_INDEX]*Gamma_f[nse.TPGA_INDEX]
        d[nse.H3_INDEX] -= A[nse.HE4_INDEX]*Gamma_r[nse.TPGA_INDEX]
        d[nse.P_INDEX] -= A[nse.H3_INDEX]*A[nse.P_INDEX]*Gamma_f[nse.TPGA_INDEX]
        d[nse.P_INDEX] += A[nse.HE4_INDEX]*Gamma_r[nse.TPGA_INDEX]
        d[nse.HE4_INDEX] += A[nse.H3_INDEX]*A[nse.P_INDEX]*Gamma_f[nse.TPGA_INDEX]
        d[nse.HE4_INDEX] -= A[nse.HE4_INDEX]*Gamma_r[nse.TPGA_INDEX]

    if (p==2):
        for i in range(9):
            
            d[fwd1[i]] -= A[fwd1[i]]*A[fwd2[i]]*Gamma_f[i]
            d[fwd1[i]] += A[rev[i]]*Gamma_r[i]
            d[fwd2[i]] -= A[fwd1[i]]*A[fwd2[i]]*Gamma_f[i]
            d[fwd2[i]] += A[rev[i]]*Gamma_r[i]
            d[rev[i]] +=A[fwd1[i]]*A[fwd2[i]]*Gamma_f[i]
            d[rev[i]] -= A[rev[i]]*Gamma_r[i]

        d = linearize(A, d, der[1], a, weak.Npntot(T,a), weak.Nnptot(T,a), Gamma_f, Gamma_r, indexes)
  
    return depvar(der[0], der[1], der[2], d*der[1])
        
        

