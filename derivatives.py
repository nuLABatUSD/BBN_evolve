import numpy as np
import thermodynamics as thermo
from constants import me, mpl
import weakrates as weak
import expansion as ex
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

#Hubble expansion rate



    

def f(a,y,p):
   
    T , t , Yp , Yn , eta = y
    der = np.zeros(4)
   
    der[0] = -(3*a**2*ex.sth(T))/(a**3*ex.dsthdT(T))
    der[1] = (1/a)*(((8*np.pi*ex.ptot(T,a))/(3*(mpl)**2)))**(-1/2)
    der[2] = (y[3]*weak.Nnptot(T,a)-y[2]*weak.Npntot(T,a))*der[1]
    der[3] = (-y[3]*weak.Nnptot(T,a)+y[2]*weak.Npntot(T,a))*der[1]
    der[4] = -3*eta*((1/T)*der[0]+(1/a))

    
    return der

