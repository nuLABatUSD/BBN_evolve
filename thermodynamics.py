import numpy as np
import matplotlib.pyplot as plt
x,w=np.polynomial.laguerre.laggauss(50)

def I1(u):                                
    g=np.zeros(50)
    for j in range(len(g)):
        E=np.sqrt(x[j]**2+u**2)           #E is epsilon
        g[j]=x[j]**2*E*(1/(np.exp(E)+1))
    return np.sum(w*g*np.exp(x))


def I2(u):
    g=np.zeros(50)
    for j in range(len(g)):
        E=np.sqrt(x[j]**2+u**2)             #E is epsilon
        g[j]=(x[j]**4)/E*(1/(np.exp(E)+1))
    return np.sum(w*g*np.exp(x))


def I3(u):
    g=np.zeros(50)
    for j in range(len(g)):
        E=np.sqrt(x[j]**2+u**2)                               #E is epsilon
        g[j]=x[j]**2*(E**2)*(np.exp(E))/(((np.exp(E)+1))**2)
    return np.sum(w*g*np.exp(x))


def I4(u):
    g=np.zeros(50)
    for j in range(len(g)):
        E=np.sqrt(x[j]**2+u**2)                               #E is epsilon
        g[j]=x[j]**4*(np.exp(E))/(np.exp(E)+1)**2
    return np.sum(w*g*np.exp(x))


def p(T,m,g):
    u=m/T
    return ((g*T**4)*(I1(u)))/(2*np.pi**2)


def P(T,m,g):
    u=m/T
    return (g*T**4)/(6*np.pi**2)*I2(u)

def dpdT(T,m,g):
    u=m/T
    return (g*T**3*I3(u))/(2*np.pi**2)


def dPdT(T,m,g):
    u=m/T
    return (g*T**3*I4(u))/(6*np.pi**2)


def s(T,m,g):
    return (1/T)*(p(T,m,g)+P(T,m,g))


def dsdT(T,m,g):
    return (1/T)*(dpdT(T,m,g)+dPdT(T,m,g))-(1/T**2)*(p(T,m,g)+P(T,m,g))