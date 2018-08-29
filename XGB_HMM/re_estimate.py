import numpy as np
from public_tool.form_index import form_index


def re_estimate(A, B_all, pi, lengths):
    # through maximizing the likelihood, evaluate translation proba and prior proba, and the best states sequence and likelihood
    # input:
    #     A, array, (n_states, n_states), transition probability among states
    #     B_all, array, (n_samples, n_states), P(0|S)
    #     pi, array, (n_states, n_states), prior probability of states
    #     lengths, list, lengths of sequence
    # output:
    #     A, array, (n_states, n_states), translation probability of n_states
    #     S, array, (n_samples, ), the best state sequence

    n_states = B_all.shape[1]
    T_all = B_all.shape[0]
    alpha_all = np.zeros((T_all, n_states))
    beta_all = np.zeros((T_all, n_states))
    di_gamma_all = np.zeros((T_all, n_states, n_states))
    gamma_all = np.zeros((T_all, n_states))
    scale_all = np.zeros(T_all)

    for k in range(len(lengths)):

        begin_index, end_index = form_index(lengths, k)
        T = end_index-begin_index
        B = B_all[begin_index:end_index].copy()
        alpha = np.zeros((T, n_states))
        beta = np.zeros((T, n_states))
        di_gamma = np.zeros((T, n_states, n_states))
        gamma = np.zeros((T, n_states))
        scale = np.zeros(T)

        # compute alpha
        # t = 0
        for i in range(n_states):
            alpha[0, i] = pi[i] * B[0, i]
        scale[0] = sum(alpha[0])
        # t = 1, 2, ..., T-1
        for t in range(1, T):
            for i in range(n_states):
                alpha[t, i] = 0
                for j in range(n_states):
                    alpha[t, i] += alpha[t-1, j] * A[j, i]
                alpha[t, i] = alpha[t, i] * B[t, i]
            scale[t] = 1/sum(alpha[t])
            alpha[t] = alpha[t] * scale[t]

        # compute beta
        # t = 0
        beta[T-1] = scale[T-1]
        # t = T-2, T-3, ..., 0
        for t in range(T-2, -1, -1):
            for i in range(n_states):
                beta[t, i] = 0
                for j in range(n_states):
                    beta[t, i] += A[i, j] * B[t+1, j] * beta[t+1, j]
                beta[t, i] = scale[t] * beta[t, i]

        # compute di_gamma and gamma
        # t = 0, 1, ..., T-2
        for t in range(T-1):
            for i in range(n_states):
                gamma[t, i] = 0
                for j in range(n_states):
                    di_gamma[t, i, j] = alpha[t, i] * A[i, j] * B[t+1, j] * beta[t+1, j]
                    gamma[t, i] += di_gamma[t, i, j]
        # t = T-1
        t = T-1
        gamma[t] = alpha[t]

        # record
        alpha_all[begin_index:end_index] = alpha
        beta_all[begin_index:end_index] = beta
        di_gamma_all[begin_index:end_index] = di_gamma
        gamma_all[begin_index:end_index] = gamma
        scale_all[begin_index:end_index] = scale

    # re-estimate A
    for i in range(n_states):
        for j in range(n_states):
            numer = 0
            denom = 0
            for k in range(len(lengths)):
                begin_index, end_index = form_index(lengths, k)
                numer += np.sum(di_gamma_all[begin_index:end_index, i, j])
                denom += np.sum(gamma_all[begin_index:end_index, i])
            A[i, j] = numer/denom

    S = np.array([np.argmax(i) for i in gamma_all])
    sample_weights = np.array([np.max(i) for i in gamma_all])

    return A, gamma_all
