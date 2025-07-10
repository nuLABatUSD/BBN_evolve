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


def f(a,y,p):
   
    T,t=y
    der = np.zeros(2)
   
    der[0] = -(3*a**2*thermo.s(10,0.511,2)/(a**3*thermo.dsdT(10,0.511,2)))
    der[1] = (1/a)*((8*np.pi*thermo.p(10,0.511,2))/(3*(1.22*10**28)**2))**(1/2)

    
    return der

