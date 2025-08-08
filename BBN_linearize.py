import numpy as np
from nseabundance import P_INDEX, N_INDEX

import numba as nb

@nb.njit()
def linearize(Ab, d_full, dtda, a, Lambda_pn, Lambda_np, x_gamma_forward_rates, x_gamma_reverse_rates, x_gamma_index):
    dt = dtda * a * 0.0001
    M = np.zeros((9,9))
    M[P_INDEX, P_INDEX] += - Lambda_pn
    M[P_INDEX, N_INDEX] += Lambda_np

    M[N_INDEX, P_INDEX] += Lambda_pn
    M[N_INDEX, N_INDEX] += - Lambda_np

    N_rates = len(x_gamma_forward_rates)

    for i in range(N_rates):
        M[x_gamma_index[i,0], x_gamma_index[i,0]] += - x_gamma_forward_rates[i] * Ab[x_gamma_index[i,1]] / 2
        M[x_gamma_index[i,0], x_gamma_index[i,1]] += - x_gamma_forward_rates[i] * Ab[x_gamma_index[i,0]] / 2
        M[x_gamma_index[i,0], x_gamma_index[i,2]] += x_gamma_reverse_rates[i]

        M[x_gamma_index[i,1], x_gamma_index[i,0]] += - x_gamma_forward_rates[i] * Ab[x_gamma_index[i,1]] / 2
        M[x_gamma_index[i,1], x_gamma_index[i,1]] += - x_gamma_forward_rates[i] * Ab[x_gamma_index[i,0]] / 2
        M[x_gamma_index[i,1], x_gamma_index[i,2]] += x_gamma_reverse_rates[i]

        M[x_gamma_index[i,2], x_gamma_index[i,0]] += x_gamma_forward_rates[i] * Ab[x_gamma_index[i,1]] / 2
        M[x_gamma_index[i,2], x_gamma_index[i,1]] += x_gamma_forward_rates[i] * Ab[x_gamma_index[i,0]] / 2
        M[x_gamma_index[i,2], x_gamma_index[i,2]] += - x_gamma_reverse_rates[i]

    D_mat = np.identity(9) - M * dt

    #print(np.linalg.inv(D_mat))
    d_lin = np.linalg.inv(D_mat) @ d_full

    return d_lin