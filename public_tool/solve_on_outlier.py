from public_tool.form_index import form_index
import numpy as np
from dataset_code.process_on_raw_data import fill_na


def solve_on_outlier(dataset, lengths):
    """
    find the outlier data, and replace then by fill_na function
    input:
        dataset, array
        lengths, list, record the length of chains
    output:
        dataset, array
    """

    n = 3     # 如果是比均值相差了n个单位的标准差，那么判断为outlier
    result = np.zeros(dataset.shape)
    for i in range(len(lengths)):
        begin_index, end_index = form_index(lengths, i)
        for j in range(dataset.shape[1]):
            temp = dataset[begin_index:end_index, j].copy()
            if max(temp) > 4.5:
                flag = 1
            mean = np.mean(temp)
            std = np.std(temp)
            temp[np.logical_or(temp >= mean+n*std, temp <= mean-n*std)] = np.mean(temp)
            # temp = fill_na(temp, 100)
            result[begin_index:end_index, j] = temp

    return result
