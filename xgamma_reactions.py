from constants import zeta3
import nseabundance as nse

Nrxn = 10

fwd1 = np.zeros(xg.Nrnx, dtype=int)
fwd2 = np.zeros(xg.Nrxn, dtype=int)
rev = np.zeros(xg.Nrxn, dtype=int)


PNGD_INDEX = 0
def pngd(T):
    T9 = T*MeVtoT9
    if T9<=1.5:
        F = 44216.*(1+3.75191*T9+1.92934*T9*T9+0.746503*T9*T9*T9+0.0197023*(T9**4)+3.00491e-6*(T9**5))/(1+5.4678*T9+5.62395*T9*T9+0.489312*T9*T9*T9+0.00747806*(T9**4))
    else:
        F = (1.-np.sqrt(T9)*0.8504+T9*0.4895-(T9**(3/2))*0.09623+T9*0.008471*T9-T9*2.8e-4*(T9**(3/2)))*47420
    return F*cmgstoMeV*mN

fwd1[PNGD_INDEX] = nse.P_INDEX
fwd2[PNGD_INDEX] = nse.N_INDEX
rev[PNGD_INDEX] = nse.H2_INDEX


TPGA_INDEX = 1
def tpga(T):
    T9 = T*MeVtoT9
    F = T9**(-2/3)*2.2e4*np.exp(-3.869/(T9**(1/3)))*((T9**(1/3))*0.108+1.+(T9**(2/3))*1.68+T9*1.26+(T9**(4/3))*0.551+(T9**(5/3))*1.06)
    return F*cmgstoMeV*mN

fwd1[TPGA_INDEX] = nse.H2_INDEX
fwd2[TPGA_INDEX] = nse.P_INDEX
rev[TPGA_INDEX] = nse.HE4_INDEX


DNGT_INDEX = 2
def dngt(T):
    T9 = T*MeVtoT9
    F = (T9*18.9+1)*66.2
    return F*cmgstoMeV*mN
fwd1[DNGT_INDEX]

HE3NGA_INDEX = 3
def he3nga(T):
    T9 = T*MeVtoT9
    F = (T9*905.+1.)*6.62
    return F*cmgstoMeV*mN

LI6NGLI7_INDEX = 4
def li6ngli7(T):
    T9 = T*MeVtoT9
    F = 5100
    return F*cmgstoMeV*mN

LI6PGBE7_INDEX = 5
def li6pgbe7(T):
    T9 = T*MeVtoT9
    F = 1.25e6*(T9**(-2/3))*np.exp(-8.415/(T9**(1/3)))*(1.-0.252*T9+5.19e-2*T9*T9-2.92e-3*T9*T9*T9)
    return F*cmgstoMeV*mN    

H2AGLI6_INDEX = 6
def h2agli6(T):
    T9 = T*MeVtoT9
    F = 1.482e1*(T9**(-2./3.))*np.exp(-7.435/(T9**(1./3.)))*(1.+6.572*T9+7.6e-2*T9*T9+2.48e-2*T9*T9*T9)+8.28e1*(T9**(-3./2.))*np.exp(-7.904/T9)
    return F*cmgstoMeV*mN   

H3AGLI7_INDEX = 7
def h3agli7(T):
    T9 = T*MeVtoT9
    if T9<2.5:
        F = ((0.094614248-4.9273133*T9+99.358965*T9*T9-989.81236*T9*T9*T9+4368.45*(T9**4.)+931.93597**(T9**5.)-391.07855*pow(T9,6.)+159.23101*pow(T9,7.)-34.407594*pow(T9,8.)+3.3919004*pow(T9,9.)+0.017556217*(T9**10.)-0.036253427**(T9**11.)+0.0031118827*(T9**12.)-0.00008714468*(T9**13.))*(T9**(-1./2.)))/(np.exp(8.4e-7*T9)*((1.+1.78616593*T9)**3.))
    else: 
        F = 807.406
    return F*cmgstoMeV*mN

HE3AGBE7_INDEX = 8
def he3agbe7(T):
    T9 = T*MeVtoT9
    if T9<=100:
        F = np.exp(15.609867-(12.82707707/(T9**(1./3.)))-0.6666667*log(T9))*((1-0.020478*(T9**(2./3.))+0.211995*(T9**(4./3.)))/(1+0.255059*(T9**(2./3.))+0.338573*(T9**(4./3.))))
    else:
        F = 0
    return F*cmgstoMeV*mN

HE3TGLI6_INDEX = 9
def he3tgli6(T):
    T9 = T*MeVtoT9
    F = .2201e6*pow(T9,-2./3.)*exp(-7.73436/pow(T9,1./3.))*(1.+5.38722e-2*pow(T9,1./3.)-.214*(1.+.377*pow(T9,1./3.))*pow(T9,2./3.)+.2733*(1.+.959*pow(T9,1./3.))*pow(T9,4./3.)-1.53e-2*(1.+.959*pow(T9,1./3.))*T9*T9)*(1.-.213646*pow(T9,2./3.)+.136643*pow(T9,4./3.)-7.65244e-3*T9*T9)
    return F*cmgstoMeV*mN


indexes = np.zeros((xg.Nrxn,3), dtype=int)
indexes[:,0] = fwd1
indexes[:,1] = fwd2
indexes[:,2] = rev

def Gamma(T,eta,Yp,Yn):
    nB = nB = eta*(3/2)*zeta3*T**3
    Gamma_f = np.zeros()
    Gamma_r = np.zeros()
    for i in range(Nrxn):
        Gamma_f[i] = 
        Gamma_r[i] = 






