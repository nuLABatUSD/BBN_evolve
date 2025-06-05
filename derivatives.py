import numpy as np

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
def f(x, y, p):
    
    A, B, C, D = y  # y is an array of length 4, and its values are A, B, C, D
    
    der = np.zeros(4) # der is an array of length 4 that has dA/dt, dB/dt, dC/dt, dD/dt
    
    der[0] = y[1]
    der[1] = y[2]
    der[2] = y[3]
    der[3] = np.pi**4 * y[0]
    
    return der