from constants import zeta3,  MeVtoT9, cmgstoMeV, mN
import nseabundance as nse
import numpy as np
from numpy import sqrt, exp
import numba as nb
import interpolate_nuclear as intnuc

Nrxn = 13

fwd1 = np.zeros(Nrxn, dtype=int)
fwd2 = np.zeros(Nrxn, dtype=int)
rev1 = np.zeros(Nrxn, dtype=int)
rev2 = np.zeros(Nrxn, dtype=int)

HE3NPT_INDEX = 0
@nb.njit()
def he3npt(T):
    T9 = T*MeVtoT9
    if T9<2.5:
        F = 7.064935e8+6.733213571736319e8*T9+1.7181155480346258e9*T9*T9-4.5367658146835446e8*T9*T9*T9-1.2216728981712557e8*pow(T9,4.)-4.92736677238425e8*sqrt(T9)-1.3659670893994067e9*pow(T9,3./2.)-6.629932739639357e8*pow(T9,5./2.)+4.834951929033479e8*pow(T9,7./2.);
    else:
        F = 4.81732e8
    return F*cmgstoMeV*mN
fwd1[HE3NPT_INDEX] = nse.HE3_INDEX
fwd2[HE3NPT_INDEX] = nse.N_INDEX
rev1[HE3NPT_INDEX] = nse.H3_INDEX
rev2[HE3NPT_INDEX] = nse.P_INDEX


BE7NPLI7_INDEX = 1
@nb.njit()
def be7npli7(T):
    T9 = T*MeVtoT9
    if T9<2.5:
        F = 6.8423032e9+1.7674863e10*T9+2.6622006e9*T9*T9-3.3561608e8*T9*T9*T9-5.9309139e6*pow(T9,4.)-1.4987996e10*sqrt(T9)-1.0576906e10*pow(T9,3./2.)+2.7447598e8*pow(T9,5./2.)+7.6425157e7*pow(T9,7./2.)-2.282944e7*pow(T9,-3./2.) / exp(0.050351813/T9);
    else:
        F = 1.28039e9
    return F*cmgstoMeV*mN
fwd1[BE7NPLI7_INDEX] = nse.BE7_INDEX
fwd2[BE7NPLI7_INDEX] = nse.N_INDEX
rev1[BE7NPLI7_INDEX] = nse.LI7_INDEX
rev2[BE7NPLI7_INDEX] = nse.P_INDEX


LI6NAT_INDEX = 2
@nb.njit()
def li6nat(T):
    T9 = T*MeVtoT9
    F = pow(T9,-1.5)*2.54e9*exp(-2.39/T9)+(1.-pow(T9/(T9*49.18+1.),1.5)*0.261/pow(T9,1.5))*1.68e8
    return F*cmgstoMeV*mN
fwd1[LI6NAT_INDEX] = nse.LI6_INDEX
fwd2[LI6NAT_INDEX] = nse.N_INDEX
rev1[LI6NAT_INDEX] = nse.HE4_INDEX
rev2[LI6NAT_INDEX] = nse.H3_INDEX


BE7NAA_INDEX = 3
@nb.njit()
def be7naa(T):
    T9 = T*MeVtoT9
    if T9<10:
        F = exp(-17.8984+0.2711/T9-23.8918*pow(T9,-1./3.)+62.2135*pow(T9,1./3.)-5.2888*T9+0.3869*pow(T9,5./3.)-22.6197*np.log(T9))
    else:
        F = 0

    return F*cmgstoMeV*mN
   
fwd1[BE7NAA_INDEX] = nse.BE7_INDEX
fwd2[BE7NAA_INDEX] = nse.N_INDEX
rev1[BE7NAA_INDEX] = nse.HE4_INDEX
rev2[BE7NAA_INDEX] = nse.HE4_INDEX


LI6PAHE3_INDEX = 4
@nb.njit()
def li6pahe3(T):
    T9 = T*MeVtoT9
    if T9<2.5:
        F = ((-8.9654123e7-2.5851582e8*T9-2.6831252e7*T9*T9+3.8691673e8*pow(T9,1./3.)+4.9721269e8*pow(T9,2./3.)+2.6444808e7*pow(T9,4./3.)-1.2946419e6*pow(T9,5./3.)-1.0941088e8*pow(T9,7./3.)+9.9899564e7*pow(T9,8./3.))*pow(T9,-2./3.))/exp(7.73389632*pow(T9,-1./3.))
        F += exp(-1.137519e0*T9*T9-8.6256687*pow(T9,-1./3.))*(3.0014189e7-1.8366119e8*T9+1.7688138e9*T9*T9-8.4772261e9*T9*T9*T9+2.0237351e10*pow(T9,4.)-1.9650068e10*pow(T9,5.)+7.9452762e8*pow(T9,6.)+1.3132468e10*pow(T9,7.)-8.209351e9*pow(T9,8.)-9.1099236e8*pow(T9,9.)+2.7814079e9*pow(T9,10.)-1.0785293e9*pow(T9,11.)+1.3993392e8*pow(T9,12.))*pow(T9,-2./3.)
    else:
        F = 1.53403e6
        F += 84516.7
    return F*cmgstoMeV*mN
fwd1[LI6PAHE3_INDEX] = nse.LI6_INDEX
fwd2[LI6PAHE3_INDEX] = nse.P_INDEX
rev1[LI6PAHE3_INDEX] = nse.HE4_INDEX
rev2[LI6PAHE3_INDEX] = nse.HE3_INDEX


LI7PAA_INDEX = 5
@nb.njit()
def li7paa(T):
    T9 = T*MeVtoT9
    if T9<2.5:
        F = ((-7.4966212e7-1.9411561e10*T9+1.6262854e10*T9*T9+2.0533495e7*pow(T9,1./3.)+3.9547491e9*pow(T9,2./3.)+3.7907358e10*pow(T9,4./3.)-3.4313768e10*pow(T9,5./3.)-3.9965228e9*pow(T9,7./3.)+4.0333873e8*pow(T9,8./3.))*pow(T9,-2./3.))/exp(4.62619323*pow(T9,-1./3.))
    else:
        F = 3.05102e7
    return F*cmgstoMeV*mN
fwd1[LI7PAA_INDEX] = nse.LI7_INDEX
fwd2[LI7PAA_INDEX] = nse.P_INDEX
rev1[LI7PAA_INDEX] = nse.HE4_INDEX
rev2[LI7PAA_INDEX] = nse.HE4_INDEX


TDNA_INDEX = 6
@nb.njit()
def tdna(T):
    T9 = T*MeVtoT9
    if T9<2.5:
        F = 6.2265733e8/(exp(0.49711597/T9)*pow(T9,0.56785403))+exp(-0.23309803*T9*T9-1.342742*pow(T9,-1./3.))*(-8.1144927e7 + 2.2315324e9*T9-2.9439669e9*T9*T9+1.8764462e9*T9*T9*T9 - 6.0511612e8*pow(T9,4.)+9.5196576e7*pow(T9,5.)-5.2901086e6*pow(T9,6.))*pow(T9,-2./3.)
    else:
        F = 3.40249e8
    return F*cmgstoMeV*mN
fwd1[TDNA_INDEX] = nse.H3_INDEX
fwd2[TDNA_INDEX] = nse.H2_INDEX
rev1[TDNA_INDEX] = nse.N_INDEX
rev2[TDNA_INDEX] = nse.HE4_INDEX


HE3DPA_INDEX = 7
@nb.njit()
def he3dpa(T):
    T9 = T*MeVtoT9
    if T9<2.5:
        F = 3.1038385e8/(exp(1.6190981/T9)*pow(T9,0.12159455))+exp(-0.0062340825*T9*T9-1.4540617*pow(T9,-1./3.))*(-3.1335916e7-6.2051071e8*T9-1.8782248e9*T9*T9+6.5642773e8*T9*T9*T9+1.530887e8*pow(T9,4.)-4.9542138e8*pow(T9,10./3.)-1.770285e8*pow(T9,11./3.)+1.14185e8*pow(T9,1./3.)-2.516526e7*pow(T9,13./3.)+1.7500204e8*pow(T9,2./3.)-1.7513362e9*pow(T9,4./3.)+5.2792247e9*pow(T9,5./3.)-3.32382e9*pow(T9,7./3.)+2.0346284e9*pow(T9,8./3.))*pow(T9,-2./3.)
    else:
        F = 1.55167e8
    return F*cmgstoMeV*mN
fwd1[HE3DPA_INDEX] = nse.HE3_INDEX
fwd2[HE3DPA_INDEX] = nse.H2_INDEX
rev1[HE3DPA_INDEX] = nse.P_INDEX
rev2[HE3DPA_INDEX] = nse.HE4_INDEX


LI6DNBE7_INDEX = 8
@nb.njit()
def li6dnbe7(T):
    T9 = T*MeVtoT9
    F = 1.48e12*pow(T9,-2./3.)*exp(-10.135/pow(T9,1./3.))
    return F*cmgstoMeV*mN
fwd1[LI6DNBE7_INDEX] = nse.LI6_INDEX
fwd2[LI6DNBE7_INDEX] = nse.H2_INDEX
rev1[LI6DNBE7_INDEX] = nse.N_INDEX
rev2[LI6DNBE7_INDEX] = nse.BE7_INDEX


HE3TDA_INDEX = 9
@nb.njit()
def he3tda(T):
    T9 = T*MeVtoT9
    F = 5.46e9*pow(T9/(1.+.128*T9),.8333333)*pow(T9,-3./2.)*exp(-7.733/(pow(T9/(1.+.128*T9),.333333)))
    return F*cmgstoMeV*mN
fwd1[HE3TDA_INDEX] = nse.HE3_INDEX
fwd2[HE3TDA_INDEX] = nse.H3_INDEX
rev1[HE3TDA_INDEX] = nse.H2_INDEX
rev2[HE3TDA_INDEX] = nse.HE4_INDEX


LI6DPLI7_INDEX = 10   # confusion about whether P or N is meant to be in products
@nb.njit()
def li6dpli7(T):
    T9 = T*MeVtoT9
    F = 1.48e12*pow(T9,-2./3.)*exp(-10.135/pow(T9,1./3.))
    return F*cmgstoMeV*mN
fwd1[LI6DPLI7_INDEX] = nse.LI6_INDEX
fwd2[LI6DPLI7_INDEX] = nse.H2_INDEX
rev1[LI6DPLI7_INDEX] = nse.P_INDEX
rev2[LI6DPLI7_INDEX] = nse.LI7_INDEX


DDNHE3_INDEX = 11
@nb.njit()
def ddnhe3(T):
    T9 = T*MeVtoT9
    #if T9<9.617518190868742:
        #F = T9**(-2/3)*np.exp(-T9**(-1/3))*(-1.84664e6+1.22986e7*T9**(1/3)-1.3761e7*T9**(2/3)-6.11628e7*T9+1.3329e8*T9**(4/3)-1.24333e7*T9**(5/3)-2.72404e7*T9**2+8.52947e6*T9**(7/3)+2.2519e6*T9**(8/3)-2.31204e6*T9**3-294342*T9**(10/3)+911550*T9**(11/3)-252211*T9**4)
    #else:
        #F = 76709397.15703112
    F = intnuc.fitting(T9, intnuc.ddnhe3_index)
    return F*cmgstoMeV*mN
fwd1[DDNHE3_INDEX] = nse.H2_INDEX
fwd2[DDNHE3_INDEX] = nse.H2_INDEX
rev1[DDNHE3_INDEX] = nse.N_INDEX
rev2[DDNHE3_INDEX] = nse.HE3_INDEX


DDPT_INDEX = 12
@nb.njit()
def ddpt(T):
    T9 = T*MeVtoT9
    #if T9<18.29674756074093:
        #F = T9**(-2/3)*np.exp(-1.06765*T9**(-1/3))*(-5.85032e6+5.23171e7*T9**(1/3)-1.70199e8*T9**(2/3)+2.32242e8*T9-1.18812e8*T9**#(4/3)+5.28874e7*T9**(5/3)-9.85542e6*T9**2)
    #else:
        #F = 85567388.72783639
    F = intnuc.fitting(T9,intnuc.ddpt_index)
    return F*cmgstoMeV*mN
fwd1[DDPT_INDEX] = nse.H2_INDEX
fwd2[DDPT_INDEX] = nse.H2_INDEX
rev1[DDPT_INDEX] = nse.P_INDEX
rev2[DDPT_INDEX] = nse.H3_INDEX


indexes_xy = np.zeros((Nrxn,4), dtype=int)
indexes_xy[:,0] = fwd1
indexes_xy[:,1] = fwd2
indexes_xy[:,2] = rev1
indexes_xy[:,3] = rev2


def Gammaxy(T,eta):
   
    n_gamma = (2/np.pi**2)*zeta3*T**3
    nB = eta*n_gamma
    #A_NSE = nse.nse(T, eta)
    Gamma_f = np.zeros(Nrxn)
    Gamma_r = np.zeros(Nrxn)    
    Gamma_f[HE3NPT_INDEX] = he3npt(T)*nB
    Gamma_f[BE7NPLI7_INDEX] = be7npli7(T)*nB
    Gamma_f[LI6NAT_INDEX] = li6nat(T)*nB
    Gamma_f[BE7NAA_INDEX] = be7naa(T)*nB
    Gamma_f[LI6PAHE3_INDEX] = li6pahe3(T)*nB
    Gamma_f[LI7PAA_INDEX] = li7paa(T)*nB
    Gamma_f[TDNA_INDEX] = tdna(T)*nB
    Gamma_f[HE3DPA_INDEX] = he3dpa(T)*nB
    Gamma_f[LI6DNBE7_INDEX] = li6dnbe7(T)*nB
    Gamma_f[HE3TDA_INDEX] = he3tda(T)*nB
    Gamma_f[LI6DPLI7_INDEX] = li6dpli7(T)*nB
    Gamma_f[DDNHE3_INDEX] = ddnhe3(T)*nB
    Gamma_f[DDPT_INDEX] = ddpt(T)*nB
    
    for i in range(Nrxn):
        deltaB = nse.B[fwd1[i]]+nse.B[fwd2[i]]-nse.B[rev2[i]]-nse.B[rev1[i]]
        Gamma_r[i] = Gamma_f[i]*((nse.g[fwd1[i]]*nse.g[fwd2[i]])/(nse.g[rev1[i]]*nse.g[rev2[i]]))*((nse.A[fwd1[i]]*nse.A[fwd2[i]])/(nse.A[rev1[i]]*nse.A[rev2[i]]))**(3/2)*np.exp(deltaB/T)
   
    return Gamma_f, Gamma_r