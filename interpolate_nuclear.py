import numpy as np
import numba as nb

data = np.load("nuclear_data.npy")
data_T9 = data[0,:]

dpghe3_index = 1
ddnhe3_index = 2
ddpt_index = 3


@nb.njit()
def find_index_below(T9):
    if T9 > data_T9[0]:
        return -999
        
    if T9 < data_T9[-1]:
        return len(data_T9) - 1

    index = np.where(data_T9 > T9)[0][-1]
    return index

@nb.njit()
def quad_int(x0, x, y):
    res = y[0] * (x0 - x[1])/(x[0] - x[1]) * (x0 - x[2])/(x[0] - x[2])
    res += y[1] * (x0 - x[0])/(x[1] - x[0]) * (x0 - x[2])/(x[1] - x[2])
    res += y[2] * (x0 - x[0])/(x[2] - x[0]) * (x0 - x[1])/(x[2] - x[1])

    return res


@nb.njit()
def fitting(T9, rxn):
    ind = find_index_below(T9)
    if ind == -999:
        return data[rxn,0]
    if ind == len(data_T9)-1:
        return data[rxn,-1]

    if ind == 0:
        x_vals = data_T9[0:3]
        y_vals = data[rxn,0:3]
    else:
        x_vals = data_T9[ind-1:ind+2]
        y_vals = data[rxn,ind-1:ind+2]

    return np.exp(quad_int(T9, x_vals, np.log(y_vals)))