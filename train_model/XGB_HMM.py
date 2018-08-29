"""
    train the XGB_HMM model, return A, B(XGB model), pi
"""
import numpy as np
from XGB_HMM.GMM_HMM import GMM_HMM
from XGB_HMM.re_estimate import re_estimate
from XGB_HMM.predict import self_pred
from XGB_HMM.xgb import self_xgb


def XGB_HMM(O, lengths, verbose=True):

    n_states = 3
    stop_flag = 0
    iteration = 1
    log_likelihood = -np.inf
    min_delta = 1e-4

    S, A, gamma = GMM_HMM(O, lengths, n_states, verbose=True)
    prior_pi = np.array([sum(S == i) / len(S) for i in range(n_states)])
    # model = self_xgb(O, gamma, n_states)
    model = 1
    # B_Matrix = form_B_matrix_by_DNN(model, O, prior_pi)
    B_Matrix = gamma / prior_pi

    record_log_likelihood = []
    best_result = []  # record the result A, B(xgb model), prior pi, best_log_likelihood

    while stop_flag <= 3:

        A, gamma = re_estimate(A, B_Matrix, prior_pi, lengths)
        # pickle.dump([O, gamma], open('C:/Users/Administrator/Desktop/HMM_program/save/temp.pkl', 'wb'))
        # model = self_xgb(O, gamma, n_states)

        # model = self_DNN(O, gamma)
        # B_Matrix = form_B_matrix_by_DNN(model, O, prior_pi)

        B_Matrix = gamma / prior_pi

        new_S, _, new_log_likelihood = self_pred(B_Matrix, lengths, A, prior_pi)

        record_log_likelihood.append(new_log_likelihood)

        if len(best_result) == 0:
            best_result = [A, model, prior_pi, new_log_likelihood]
        elif new_log_likelihood > best_result[3]:
            best_result = [A, model, prior_pi, new_log_likelihood]
            temp = gamma

        if new_log_likelihood - log_likelihood <= min_delta:
            stop_flag += 1
        else:
            stop_flag = 0

        log_likelihood = new_log_likelihood
        iteration += 1

        if verbose:
            print(new_log_likelihood)

    model = self_xgb(O, temp, n_states)
    best_result[1] = model

    return best_result[0], best_result[1], best_result[2]
