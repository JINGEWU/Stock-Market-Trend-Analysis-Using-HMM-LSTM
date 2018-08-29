import matplotlib.pyplot as plt
import numpy as np


def form_color_dict(S, label):
    """
    指定画图的颜色，找到3个state中最代表上升的为红色，最代表下降的为绿色，中间的为蓝色
    input:
        state_index, indicate the index of the state
        S, array, (n_samples, ), the state sequence
        label, array, (n_samples, ), the label sequence
    output:
        true, dict, transform state index to str color
    """
    n_states = len(np.unique(S))
    count = np.zeros(n_states)
    record = np.zeros((n_states, 3))

    S = np.delete(S, np.where(label == -2), 0)
    label = np.delete(label, np.where(label == -2), 0)

    for i in range(len(S)):
        temp = int(S[i])
        count[temp] += 1
        record[temp, int(label[i])+1] += 1

    for i in range(n_states):
        record[i] = record[i]/count[i]

    blue_index = np.argmax(record[:, 1])

    else_index = [0, 1, 2]
    else_index.remove(int(blue_index))

    index1 = else_index[0]
    index2 = else_index[1]
    flag1 = record[index1, 2] - record[index1, 0]
    flag2 = record[index2, 2] - record[index2, 0]
    if np.abs(flag1) > np.abs(flag2):
        if flag1 > 0:
            red_index = index1
            green_index = index2
        else:
            green_index = index1
            red_index = index2
    else:
        if flag2 > 0:
            red_index = index2
            green_index = index1
        else:
            green_index = index2
            red_index = index1

    print(record)

    true = dict(zip([green_index, red_index, blue_index], ['g', 'r', 'b']))

    return true


def plot_result(price, label, S, new_S, n_states, record_log_likelihood, record_plot_score):
    # visualize the result
    # 1- the state plot formed by GMM-HMM
    # 2- the state plot formed by xgb-HMM
    # 3- the plot of log_likelihood and plot_score on iteration
    # input:
    #    price, array, (n_samples, ), the close price sequence
    #    label, array, (n_samples, ), the label sequence
    #    S, array, (n_samples, ), the state sequence formed by GMM-HMM
    #    new_S, array, (n_samples, ), the state sequence formed by xgb-HMM
    #    n_states, number of states
    #    record_log_likelihood, list, record the log likelihood ordered by iteration
    #    record_plot_score, list, record the log likelihood ordered by iteration
    # output:
    #    true, dict, transform state index to str color

    plt.figure(1, figsize=(100, 50))
    # plt.rcParams['figure.dpi'] = 1000
    # plt.rcParams['savefig.dpi'] = 1000

    show_x = np.array([i for i in range(len(S))])

    plt.subplot(212)
    plt.title('XGB-HMM', size=15)
    plt.ylabel('price', size=15)
    plt.xlabel('time', size=15)
    _ = form_color_dict(S, label)
    true = form_color_dict(new_S, label)

    for i in range(n_states):
        color = true[i]
        temp = (new_S == i)
        plt.scatter(show_x[temp], price[temp], label='hidden state %d' % i, marker='o', s=10, c=color)
        plt.legend(loc='upper right')
    plt.plot(price, c='yellow', linewidth=0.5)

    plt.subplot(211)
    plt.title('GMM-HMM', size=15)
    plt.ylabel('price', size=15)
    plt.xlabel('time', size=15)

    for i in range(n_states):
        color = true[i]
        temp = (S == i)
        plt.scatter(show_x[temp], price[temp], label='hidden state %d' % i, marker='o', s=10, c=color)
        plt.legend(loc='upper right')
    plt.plot(price, c='yellow', linewidth=0.5)

    # plot log_likelihood and plot_score
    fig = plt.figure(2, (400, 400))
    ax1 = fig.add_subplot(111)
    ax1.plot(np.array([np.nan] + record_log_likelihood), label='log likelihood', c='r')
    ax1.set_ylabel('log likelihood', size=20)
    ax1.set_xlabel('iteration', size=20)
    plt.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(np.array(record_plot_score), label='plot score', c='b')
    ax2.set_ylabel('plot score', size=20)

    plt.legend(loc='upper right')

    index = np.argmax(np.array(record_log_likelihood))
    ax1.scatter(np.array(index+1), np.array(record_log_likelihood[index]), marker='o', c='k', s=60)
    ax1.annotate('best iteration', size=20, xy=(index+1, record_log_likelihood[index]),
                 xytext=(index-5, record_log_likelihood[index]*1.05), arrowprops=dict(facecolor='black', shrink=1))

    plt.show()

    return true
