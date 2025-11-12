import numpy as np
import numba as nb


data = np.load("RMFK_results.npy")
data_Tcm = data[0,:]

DQDA_INDEX = 1
RHONU_FACTOR_INDEX = 2
NP_RATE_INDEX = 3
PN_RATE_INDEX = 4

@nb.njit()
def find_index_below(Tcm):
    if Tcm > data_Tcm[0]:
        return -999
        
    if Tcm < data_Tcm[-1]:
        return len(data_Tcm) - 1

    index = np.where(data_Tcm > Tcm)[0][-1]
    return index

@nb.njit()
def quad_int(x0, x, y):
    res = y[0] * (x0 - x[1])/(x[0] - x[1]) * (x0 - x[2])/(x[0] - x[2])
    res += y[1] * (x0 - x[0])/(x[1] - x[0]) * (x0 - x[2])/(x[1] - x[2])
    res += y[2] * (x0 - x[0])/(x[2] - x[0]) * (x0 - x[1])/(x[2] - x[1])

    return res


@nb.njit()
def fitting(Tcm, rxn):
    ind = find_index_below(Tcm)
    if ind == -999:
        return data[rxn,0]
    if ind == len(data_Tcm)-1:
        return data[rxn,-1]

    if ind == 0:
        x_vals = data_Tcm[0:3]
        y_vals = data[rxn,0:3]
    else:
        x_vals = data_Tcm[ind-1:ind+2]
        y_vals = data[rxn,ind-1:ind+2]

    if y_vals[-1] == 0:
        return 0

    if rxn == RHONU_FACTOR_INDEX:
        res = (quad_int(Tcm, x_vals, (y_vals)))
    else:
        res = np.exp(quad_int(Tcm, x_vals, np.log(y_vals)))

    return res

