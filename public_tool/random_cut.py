import random


def random_cut(X, y, k_fold=5):
    # 随机切片，用于CV
    # output:
    #     X_train, y_train, X_valid, y_valid

    temp = [i for i in range(X.shape[0])]
    random.shuffle(temp)
    temp_index = int(len(temp) / k_fold)
    X_train, y_train = X[temp_index:], y[temp_index:]
    X_valid, y_valid = X[:temp_index], y[:temp_index]

    return X_train, y_train, X_valid, y_valid
