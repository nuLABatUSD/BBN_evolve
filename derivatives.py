
import numpy as np
import thermodynamics as thermo
from constants import me, mpl, zeta3, MeVtoT9, cmgstoMeV, mN
import weakrates as weak
import expansion as ex
import nseabundance as nse
from BBN_linearize import linearize
import xgamma_reactions as xg
import xy_reactions as xy
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

import decay_results as dec

def dQda_RMFK(a, Q0=0, a0=1):
    return dec.fitting(1/a, dec.DQDA_INDEX) * a**3

def dQda(a, Q0 = 0, a0 = 1):
    if Q0 == 0:
        return 0
    else:
        return Q0*a**3*np.exp(-a/a0)

def delta_nu_RMFK(T, a):
    return dec.fitting(1/a, dec.RHONU_FACTOR_INDEX)

def delta_neutrino_energy_density(a, rhonu=0):
    return 0

def weak_np_RMFK(T, a):
    return dec.fitting(1/a, dec.NP_RATE_INDEX) + weak.N(T, a)

def weak_pn_RMFK(T, a):
    return dec.fitting(1/a, dec.PN_RATE_INDEX)

def weak_np(T, a, npfact=1.):
    return weak.Nnptot(T,a) * npfact

def weak_pn(T, a, pnfact=1.):
    return weak.Npntot(T,a) * pnfact

def sep(z):
    return z[0], z[1], z[2], z[3:]
T,t,eta,A = sep(np.arange(12))


#creates array of 12 dependent variables
def depvar(T,t,eta,A):
    return np.concatenate((np.array([T,t,eta]), A))



def f(a,y,p):
    T, t, eta, A = sep(y)

    if np.isscalar(p):
        p0 = p
    else:
        p0 = p[0]
        if p0 // 2 == 1:
            p0, Q0, a0, np_factor, pn_factor, rhonu = p
  
    d=np.zeros(9)
    der=np.zeros(3)

    dq = 0
    drho = 0
    Gnp = 0
    Gpn = 0
    if p0 // 2 == 0:
        Gnp = weak_np(T, a)
        Gpn = weak_pn(T, a)
    elif p0 // 2 == 1:
        dq = dQda(a, Q0, a0)
        drho = delta_neutrino_energy_density(a, rhonu)
        Gnp = weak_np(T, a, np_factor)
        Gpn = weak_pn(T, a, pn_factor)
    elif p0 // 2 == 2:
        dq = dQda_RMFK(a)
        drho = delta_nu_RMFK(T, a)
        Gnp = weak_np_RMFK(T, a)
        Gpn = weak_pn_RMFK(T, a)

    der[0] = ((1/T)*(dq)-3*a**2*ex.sth(T))/(a**3*ex.dsthdT(T))
    der[1] = (1/a)*((8*np.pi*(ex.ptot(T,a)+drho))/(3*(mpl)**2))**(-1/2)
    der[2] = -3*eta*((1/T)*der[0]+(1/a))

    nprate = A[nse.N_INDEX]*Gnp - A[nse.P_INDEX]*Gpn
    d[nse.P_INDEX] += nprate
    d[nse.N_INDEX] -= nprate
    d[nse.H3_INDEX] = 0
    d[nse.HE3_INDEX] = 0
    d[nse.HE4_INDEX] = 0
    d[nse.BE7_INDEX] = 0
    d[nse.LI7_INDEX] = 0
    d[nse.LI6_INDEX] = 0

    if (p0 % 2 == 1):
        Gamma_f_xg, Gamma_r_xg = xg.Gammaxg(T, eta)
        Gamma_f_xy, Gamma_r_xy = xy.Gammaxy(T, eta)
      
        for i in range(xg.Nrxn):
            rxnrate = A[xg.rev[i]]*Gamma_r_xg[i] - A[xg.fwd1[i]]*A[xg.fwd2[i]]*Gamma_f_xg[i]
            
            d[xg.fwd1[i]] += rxnrate
            d[xg.fwd2[i]] += rxnrate
            d[xg.rev[i]] -= rxnrate
        
        for j in range(xy.Nrxn):
            rxnrate = A[xy.rev1[j]]*A[xy.rev2[j]]*Gamma_r_xy[j] - A[xy.fwd1[j]]*A[xy.fwd2[j]]*Gamma_f_xy[j]
            
            d[xy.fwd1[j]] += rxnrate
            d[xy.fwd2[j]] += rxnrate
            d[xy.rev1[j]] -= rxnrate
            d[xy.rev2[j]] -= rxnrate
        
        d = linearize(A, d, der[1], a, Gpn, Gnp, Gamma_f_xg, Gamma_r_xg, xg.indexes_xg, Gamma_f_xy, Gamma_r_xy, xy.indexes_xy)
  
    return depvar(der[0], der[1], der[2], d*der[1])
        
        

