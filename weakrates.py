import numpy as np
import matplotlib.pyplot as plt
x, w = np.polynomial.laguerre.laggauss(50)
xj, wj = np.polynomial.legendre.leggauss(50)
import ODESolve_slow as ODE
from constants import GF, gA, me, hbar, dmnp, mpl

#antineutrino distribution function
def fanti(E,Tcm):
    fanti=(np.exp(E/Tcm)+1)**(-1)
    return fanti

#electron distribution fucntion
def fe(E,T):
    fe=(np.exp(E/T)+1)**(-1)
    return fe

#neutrino distributuon fucntion
def fneutrino(E,Tcm):
    fneutrino=(np.exp(E/Tcm)+1)**(-1)
    return fneutrino

#positron distribution fucntion
def fpos(E,T):
    fpos=(np.exp(E/T)+1)**(-1)
    return fpos



#integrand for neutrond ecay rate
def g(x,T,Tcm):
    delta=dmnp/Tcm
    u=me/Tcm
    g=(x**2)*(delta-x)*np.sqrt((delta-x)**2-u**2)*(1-fanti(x*Tcm,Tcm))*(1-fe((delta-x)*Tcm,T))
    return g

#integrand for rate antineutrino,electron,proton
def g_antiep(x,T,Tcm):
    delta=dmnp/Tcm
    u=me/Tcm
    g_antiep = x**2*(delta-x)*np.sqrt((delta-x)**2-u**2)*fanti(x*Tcm,Tcm)*fe((delta-x)*Tcm,T)
    return g_antiep

#integrand for rate neutrino,electron,neutron
def g_neutrinoen(x,T,Tcm):
    delta=dmnp/Tcm
    u=me/Tcm
    g_neutrinoen=x**2*(x+delta)*np.sqrt((x+delta)**2-u**2)*fneutrino(x*Tcm,Tcm)*(1-fe((x+delta)*Tcm,T))
    return g_neutrinoen

#integrand for rate for electron,proton
def g_ep(x,T,Tcm):
    delta=dmnp/Tcm
    u=me/Tcm
    g_ep=x**2*(x+delta)*np.sqrt((x+delta)**2-u**2)*(1-fneutrino(x*Tcm,Tcm))*fe((x+delta)*Tcm,T)
    return g_ep

#integrand for rate for position, neutron
def g_posn(x,T,Tcm):
    delta=dmnp/Tcm
    u=me/Tcm
    g_posn=x**2*(x-delta)*np.sqrt((x-delta)**2-u**2)*(1-fanti(x*Tcm,Tcm))*(fpos((x-delta)*Tcm,T))
    return g_posn

#integrand for rate for antineutron, positron
def g_antip(x,T,Tcm):
    delta=dmnp/Tcm
    u=me/Tcm
    g_antip=x**2*(x-delta)*np.sqrt((x-delta)**2-u**2)*(fanti(x*Tcm,Tcm))*(1-fpos((x-delta)*Tcm,T))
    return g_antip




#Neutron Decay Rate
def N(T,a):
    Tcm=1/a
    delta=dmnp/Tcm
    u=me/Tcm
    integrand=np.zeros(50)
    A=delta-u
  
    for j in range(len(integrand)):
        E=(A/2)*(xj[j]+1)
        integrand[j] = g(E,T,Tcm)
    return ((GF**2)/(2*np.pi**3))*(1+(3*(gA**2)))*(Tcm**5)*(A/2)*np.sum(wj*integrand)  

#antineutrino,electron,proton Decay Rate
def N_antiep(T,a):
    Tcm=1/a
    delta=dmnp/Tcm
    u=me/Tcm
    integrand=np.zeros(50)
    A=delta-u
  
    for j in range(len(integrand)):
        E=(A/2)*(xj[j]+1)
        integrand[j] = g_antiep(E,T,Tcm)
    return ((GF**2)/(2*np.pi**3))*(1+(3*(gA**2)))*(Tcm**5)*(A/2)*np.sum(wj*integrand)  

#neutrino,electron,neutron Decay Rate
def N_neutrinoen(T,a):
    Tcm=1/a
    delta=dmnp/Tcm
    B=0

    integrand=np.zeros(50)
  
    for j in range(len(integrand)):
        E=(x[j]+B)
        u=E-B
        integrand[j] = g_neutrinoen(E,T,Tcm)
    return ((GF**2)/(2*np.pi**3))*(1+(3*(gA**2)))*(Tcm**5)*np.sum(w*integrand*np.exp(x))

#electron,proton Decay Rate
def N_ep(T,a):
    Tcm=1/a
    delta=dmnp/Tcm
    B=0
    
    integrand=np.zeros(50)
  
    for j in range(len(integrand)):
        E=(x[j]+B)
        u=E-B
        integrand[j] = g_ep(E,T,Tcm)
    return ((GF**2)/(2*np.pi**3))*(1+(3*(gA**2)))*(Tcm**5)*np.sum(w*integrand*np.exp(x))  

#positron, neutron decay rate
def N_posn(T,a):
    Tcm=1/a
    delta=dmnp/Tcm
    u=me/Tcm
    B=delta+u

    integrand=np.zeros(50)

    for j in range(len(integrand)):
        E=(x[j]+B)
        u=E-B
        integrand[j] = g_posn(E,T,Tcm)
    return ((GF**2)/(2*np.pi**3))*(1+(3*(gA**2)))*(Tcm**5)*np.sum(w*integrand*np.exp(x)) 

#antineutrino, proton decay rate
def N_antip(T,a):
    Tcm=1/a
    delta=dmnp/Tcm
    u=me/Tcm
    B=delta+u

    integrand=np.zeros(50)

    for j in range(len(integrand)):
        E=(x[j]+B)
        u=E-B
        integrand[j]=g_antip(E,T,Tcm)
    return ((GF**2)/(2*np.pi**3))*(1+(3*(gA**2)))*(Tcm**5)*np.sum(w*integrand*np.exp(x)) 






#total neutron to proton rate
def Nnptot(T,a):
    Nnptot = N_posn(T,a) + N_neutrinoen(T,a) + N(T,a)
    return Nnptot

#total proton to neutron rate
def Npntot(T,a):
    Npntot = N_antiep(T,a) + N_ep(T,a) + N_antip(T,a)
    return Npntot
    