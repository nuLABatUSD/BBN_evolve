import numpy as np
import thermodynamics as thermo
from constants import me, mpl, zeta3, MeVtoT9, cmgstoMeV, mN
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

def sep(z):
    return z[0], z[1], z[2], z[3:]
T,t,eta,A = sep(np.arange(12))

#creates array of 12 dependent variables
def depvar(T,t,eta,A):
    return np.concatenate((np.array([T,t,eta]), A))

def pngd(T):
    T9 = T*MeVtoT9
    if T9<=1.5:
        F = 44216.*(1+3.75191*T9+1.92934*T9*T9+0.746503*T9*T9*T9+0.0197023*(T9**4)+3.00491e-6*(T9**5))/(1+5.4678*T9+5.62395*T9*T9+0.489312*T9*T9*T9+0.00747806*(T9**4))
    else:
        F = (1.-np.sqrt(T9)*0.8504+T9*0.4895-(T9**(3/2))*0.09623+T9*0.008471*T9-T9*2.8e-4*(T9**(3/2)))*47420
    return F*cmgstoMeV*mN

def tpga(T):
    T9 = T*MeVtoT9
    F = T9**(-2/3)*2.2e4*np.exp(-3.869/(T9**(1/3)))*((T9**(1/3))*0.108+1.+(T9**(2/3))*1.68+T9*1.26+(T9**(4/3))*0.551+(T9**(5/3))*1.06)

    return F*cmgstoMeV*mN

def f(a,y,p):
    T, t, eta, A = sep(y)
    nB = eta*(3/2)*zeta3*T**3
    Gamma_f=np.zeros(2)
    Gamma_f[nse.PNGD_INDEX] = pngd(T)*nB
    Gamma_f[nse.TPGA_INDEX] = tpga(T)*nB
    A_NSE = nse.nse(T, eta, A[nse.P_INDEX], A[nse.N_INDEX])
    Gamma_r=np.zeros(2)
    Gamma_r[nse.PNGD_INDEX] = Gamma_f[nse.PNGD_INDEX]*((A_NSE[nse.P_INDEX]*A_NSE[nse.N_INDEX])/A_NSE[nse.H2_INDEX])
    Gamma_r[nse.TPGA_INDEX] = Gamma_f[nse.TPGA_INDEX]*((A_NSE[nse.P_INDEX]*A_NSE[nse.N_INDEX])/A_NSE[nse.H2_INDEX])
  
    der=np.zeros(3)
    d=np.zeros(9)

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
        d[nse.N_INDEX] -= A[nse.P_INDEX]*A[nse.N_INDEX]*Gamma_f[nse.PNGD_IDEX]
        d[nse.N_INDEX] += A[nse.H2_INDEX]*Gamma_r[nse.PNGD_INDEX]
        d[nse.H2_INDEX] += A[nse.P_INDEX]*A[nse.N_INDEX]*Gamma_f[nse.PNGD_INDEX]
        d[nse.H2_INDEX] -= A[nse.H2_INDEX]*Gamma_r[nse.PNGD_INDEX]
        

    return depvar(der[0], der[1], der[2], d*der[1])
        

