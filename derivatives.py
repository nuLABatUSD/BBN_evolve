import numpy as np
import thermodynamics as thermo
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

def sth(T):
    sth=((2*np.pi**2*2*T**3)/45)+thermo.s(T,0.511,2)+thermo.s(T,0.511,2)
    return sth


def dsthdT(T):
    dsthdT=(6*np.pi**2*2*T**2)/45+thermo.dsdT(T,0.511,2)+thermo.dsdT(T,0.511,2)
    return dsthdT


def ptot(T,a):
    ptot=(np.pi**2*2*T**4)/30+thermo.p(T,0.511,2)+thermo.p(T,0.511,2)+thermo.p(1/a,0,6)
    return ptot
    

def f(a,y,p):
   
    T,t=y
    der = np.zeros(2)
   
    der[0] = -(3*a**2*sth(T))/(a**3*dsthdT(T))
    der[1] = (1/a)*((8*np.pi*ptot(T,a))/(3*(1.22*10**22)**2))**(1/2)

    
    return der

