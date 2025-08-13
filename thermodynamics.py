import numpy as np
import matplotlib.pyplot as plt
import numba as nb
x,w=np.polynomial.laguerre.laggauss(50)

@nb.njit()
def I1(u):                                
    g=np.zeros(50)
    for j in range(len(g)):
        E=np.sqrt(x[j]**2+u**2)           #E is epsilon
        g[j]=x[j]**2*E*(1/(np.exp(E)+1))
    return np.sum(w*g*np.exp(x))

@nb.njit()
def I2(u):
    g=np.zeros(50)
    for j in range(len(g)):
        E=np.sqrt(x[j]**2+u**2)             #E is epsilon
        g[j]=(x[j]**4)/E*(1/(np.exp(E)+1))
    return np.sum(w*g*np.exp(x))

@nb.njit()
def I3(u):
    g=np.zeros(50)
    for j in range(len(g)):
        E=np.sqrt(x[j]**2+u**2)                               #E is epsilon
        g[j]=x[j]**2*(E**2)*(np.exp(E))/(((np.exp(E)+1))**2)
    return np.sum(w*g*np.exp(x))

@nb.njit()
def I4(u):
    g=np.zeros(50)
    for j in range(len(g)):
        E=np.sqrt(x[j]**2+u**2)                               #E is epsilon
        g[j]=x[j]**4*(np.exp(E))/(np.exp(E)+1)**2
    return np.sum(w*g*np.exp(x))

@nb.njit()
def p(T,m,g):
    u=m/T
    return ((g*T**4)*(I1(u)))/(2*np.pi**2)

@nb.njit()
def P(T,m,g):
    u=m/T
    return (g*T**4)/(6*np.pi**2)*I2(u)

@nb.njit()
def dpdT(T,m,g):
    u=m/T
    return (g*T**3*I3(u))/(2*np.pi**2)

@nb.njit()
def dPdT(T,m,g):
    u=m/T
    return (g*T**3*I4(u))/(6*np.pi**2)

@nb.njit()
def s(T,m,g):
    return (1/T)*(p(T,m,g)+P(T,m,g))

@nb.njit()
def dsdT(T,m,g):
    return (1/T)*(dpdT(T,m,g)+dPdT(T,m,g))-(1/T**2)*(p(T,m,g)+P(T,m,g))