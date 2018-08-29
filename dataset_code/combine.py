import sys
import numpy as np
from public_tool.form_index import form_index


def combine(X1, X2, allow_flag1, allow_flag2, label, lengths):
    # 由之前的数据来合成可以用在后面的数据
    # 两个X合并
    # allow_flag中是0的样本去掉，label中是-2的样本去掉
    # input:
    #     X: 要合并的数据，格式可以为1个array，也可以为多个array组合的一个list
    #     allow_flag: 代表是否可用的标记，格式可以为1个array，也可以为多个array组合的一个list，与上面的X对应
    #     label: label
    #     lengths: lengths
    # output:
    #     result_X: 新的X矩阵，array类型
    #     result_label: 新的label
    #     result_lengths：新的lengths

    if not (type(X1) == type(allow_flag1) or type(X2) == type(allow_flag2)):
        sys.exit('x 和 allow_flag的输入格式不一致')

    list_flag1 = type(X1) == list
    list_flag2 = type(X2) == list

    X = np.zeros((len(label), 0))
    allow_flag = np.zeros(len(label))
    count = 0

    if list_flag1 == 1:
        for i in range(len(X1)):
            X = np.column_stack(X, X1[i])
            allow_flag += allow_flag1[i]
            count += 1
    else:
        X = np.column_stack((X, X1))
        allow_flag += allow_flag1
        count += 1
    if list_flag2 == 1:
        for i in range(len(X2)):
            X = np.column_stack((X, X2[i]))
            allow_flag += allow_flag2[i]
            count += 1
    else:
        X = np.column_stack((X, X2))
        allow_flag += allow_flag2
        count += 1
    allow_flag[allow_flag < count] = 0
    allow_flag[allow_flag == count] = 1

    result_X = np.zeros((0, X.shape[1]))
    result_label = np.zeros(0)
    result_lengths = []

    for i in range(len(lengths)):
        begin_index, end_index = form_index(lengths, i)

        now_X = X[begin_index:end_index]
        now_allow_flag = allow_flag[begin_index:end_index]
        now_label = label[begin_index:end_index]

        temp = np.logical_and(now_allow_flag == 1, now_label != -2)

        result_X = np.row_stack((result_X, now_X[temp]))
        result_label = np.hstack((result_label, now_label[temp]))
        result_lengths.append(sum(temp))

    return result_X, result_label, result_lengths
