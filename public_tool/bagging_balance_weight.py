import numpy as np
import random


def bagging_balance_weight(X, y):
    # 解决分类数据集中类别不平衡的数据，bagging出新的数据
    # input:
    #         X, y
    #         这里输入的X可以是二维或者三维，只要保证第一维是sample_num
    #         y可以是列向量，也可以是one_hot编码的多列矩阵
    # output:
    #         X_result, y_result
    #         输出的格式和输入的X, y相同

    drop_th = 0.01  # 当某一类ratio小于这个阈值时，认为没有该类
    max_subsample_ratio = 1  # 首先记录在原来数据集里面最大的max_n_subsample，然后bagging的数据集的每一类的数量都是max_n_subsample*该参数

    if y.ndim == 1:
        y_label = y
    else:
        y_label = np.zeros(y.shape[0])
        for i in range(y.shape[0]):
            y_label[i] = np.where(y[i] == 1)[0][0]

    unique = np.unique(y_label)
    num_class = len(unique)
    unique_ratio = np.zeros(num_class)
    for i in range(num_class):
        unique_ratio[i] = sum(y_label == unique[i]) / len(y_label)

    unique_ratio[unique_ratio < drop_th] = 0

    n_bagging = int(max(unique_ratio) * len(y) * max_subsample_ratio)

    X_result = []
    y_result = []
    for i in range(num_class):
        if unique_ratio[i] == 0:
            continue
        else:
            sub_X = X[y_label == unique[i]]
            sub_y = y[y_label == unique[i]]
            for j in range(n_bagging):
                index = random.randint(0, sub_X.shape[0] - 1)
                X_result.append(sub_X[index])
                y_result.append(sub_y[index])
    X_result = np.array(X_result)
    y_result = np.array(y_result)
    # 打乱下顺序
    temp = [i for i in range(X_result.shape[0])]
    random.shuffle(temp)
    X_result = X_result[temp]
    y_result = y_result[temp]

    return X_result, y_result
