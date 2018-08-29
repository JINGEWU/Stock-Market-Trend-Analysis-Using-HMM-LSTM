import numpy as np


def evaluate_plot(state, n_states, label, lengths):
    # input:
    #     state: array of state, (n, ), 0 or 1 or 2, ...
    #     n_sates: number of states
    #     label: array of label, (n, ), -2, -1, 0, 1
    #     lengths: record of length of chains
    # output:
    #     score of the plot, whether this result is good or bad

    accuracy = np.zeros(n_states)
    entropy = np.zeros(n_states)
    count = np.zeros(n_states)
    record = np.zeros((n_states, 3))

    for i in range(len(lengths)):
        begin_index = sum(lengths[:i])
        end_index = begin_index + lengths[i]

        now_state = state[begin_index:end_index]
        now_label = label[begin_index:end_index]

        now_state = np.delete(now_state, np.where(now_label == -2), 0)
        now_label = np.delete(now_label, np.where(now_label == -2), 0)

        for j in range(len(now_state)):
            temp = int(now_state[j])
            count[temp] += 1
            record[temp, int(now_label[j]) + 1] += 1

    for i in range(n_states):
        if count[i] == 0:
            continue
        temp = record[i, :] / count[i]
        accuracy[i] = np.max(temp)
        temp[np.where(temp == 0)] = 0.001
        temp[np.where(temp == 1)] = 0.999
        entropy[i] = -sum(np.log(temp) * temp)

    temp1 = 1 / (1 + entropy)
    temp2 = count / sum(count)

    score = sum(temp1 * accuracy * temp2)

    # print('acc:', accuracy)
    # print('shang:', entropy)
    # print('ratio:', count/sum(count))

    return score
