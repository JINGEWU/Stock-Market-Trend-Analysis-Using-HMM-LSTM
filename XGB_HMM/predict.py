import numpy as np
from public_tool.form_index import form_index


def self_pred(B, lengths, A, pi):
    # through using the fixed model parameter,
    # calculate the best state sequence and the corresponding probability, and the best cal_likelihood
    # input:
    #     B, array, (n_samples, n_states), P(0|S)
    #     lengths, list, lengths of sequence
    #     A, array, (n_states, n_states), translation probability of n_states
    #     pi, array, (n_states, ), prior probability
    # output:
    #     state, array, (n_samples, ), the state sequence formed by gamma
    #     state_proba, array, (n_sample, n_states), the probability of state sequence
    #     log_likelihood, float, the best log likelihood

    log_likelihood_list = []
    n_states = len(pi)
    init_flag = 1

    for i in range(len(lengths)):
        begin_index, end_index = form_index(lengths, i)
        now_B = B[begin_index:end_index].copy()
        now_state = np.zeros(lengths[i])
        now_state_proba = np.zeros((lengths[i], n_states))

        for j in range(lengths[i]):

            if j == 0:
                now_state_proba[j] = now_B[j] * pi
            else:
                for k in range(n_states):
                    temp = now_state_proba[j-1] * A[:, k] * now_B[j, k]
                    now_state_proba[j, k] = max(temp)
            now_state_proba[j] = now_state_proba[j] / np.sum(now_state_proba[j])
            now_state[j] = np.argmax(now_state_proba[j])

        for j in range(lengths[i]):
            if j == 0:
                now_log_likelihood = np.log(pi[int(now_state[j])]) + np.log(now_B[j, int(now_state[j])])
            else:
                now_log_likelihood += np.log(A[int(now_state[j-1]), int(now_state[j])]) + np.log(now_B[j, int(now_state[j])])

        if init_flag == 1:
            state = now_state
            state_proba = now_state_proba
            init_flag = 0
        else:
            state = np.hstack((state, now_state))
            state_proba = np.row_stack((state_proba, now_state_proba))

        log_likelihood_list.append(now_log_likelihood)

    log_likelihood = 0
    for i in log_likelihood_list:
        log_likelihood += i

    return state, state_proba, log_likelihood
