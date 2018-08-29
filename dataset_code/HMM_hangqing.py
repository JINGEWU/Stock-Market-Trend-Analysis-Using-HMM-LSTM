import numpy as np
from public_tool.combine_allow_flag import combine_allow_flag
from public_tool.form_index import form_index
import warnings
warnings.filterwarnings("ignore")


def solve1(dataset, lengths, feature_col):
    # 根据dataset的样本量，形成相同样本量，排序一样的处理结果result，但是有一些是不一定能用的
    # 所以同时返回一个记录数据是否处理过的allow_flag
    # input:
    #     dataset, feature_col
    #     lengths
    # output:
    #     result: 处理后的数据
    #     allow_flag: 记录该数据是否可以使用

    result = np.zeros((dataset.shape[0], 3))
    allow_flag = np.zeros(dataset.shape[0])

    for i in range(len(lengths)):
        begin_index, end_index = form_index(lengths, i)

        closePrice = dataset[begin_index:end_index, feature_col.index('closePrice')]
        vol = dataset[begin_index:end_index, feature_col.index('turnoverVol')]
        highestPrice = dataset[begin_index:end_index, feature_col.index('highestPrice')]
        lowestPrice = dataset[begin_index:end_index, feature_col.index('lowestPrice')]

        logDel = np.log(highestPrice) - np.log(lowestPrice)
        logRet_5 = np.log(closePrice[5:]) - np.log(closePrice[:-5])
        logVol_5 = np.log(vol[5:]) - np.log(vol[:-5])

        logRet_5 = np.hstack((np.zeros(5), logRet_5))
        logVol_5 = np.hstack((np.zeros(5), logVol_5))

        temp = np.column_stack((logDel, logRet_5, logVol_5))
        result[begin_index:end_index, :] = temp

        temp = np.hstack((np.zeros(5), np.ones(end_index - begin_index - 5)))
        allow_flag[begin_index:end_index] = temp

    return result, allow_flag


def solve2(dataset, feature_col):
    # 根据dataset的样本量，形成相同样本量，排序一样的处理结果result，但是有一些是不一定能用的
    # 所以同时返回一个记录数据是否处理过的allow_flag
    # input:
    #     dataset, feature_col
    #     lengths
    # output:
    #     result: 处理后的数据
    #     allow_flag: 记录该数据是否可以使用

    result = np.zeros((dataset.shape[0], 4))
    result[:, 0] = dataset[:, feature_col.index('openPrice')]/dataset[:, feature_col.index('preClosePrice')]
    result[:, 1] = dataset[:, feature_col.index('lowestPrice')]/dataset[:, feature_col.index('preClosePrice')]
    result[:, 2] = dataset[:, feature_col.index('highestPrice')]/dataset[:, feature_col.index('preClosePrice')]
    result[:, 3] = dataset[:, feature_col.index('closePrice')]/dataset[:, feature_col.index('preClosePrice')]

    allow_flag = np.ones(result.shape[0])

    return result, allow_flag


def solve_on_raw_data(dataset, lengths, feature_col):
    # 根据dataset的样本量，形成相同样本量，排序一样的处理结果result，但是有一些是不一定能用的
    # 所以同时返回一个记录数据是否处理过的allow_flag
    # input:
    #     dataset, feature_col
    #     lengths
    # output:
    #     result: 处理后的数据
    #     allow_flag: 记录该数据是否可以使用

    result1, allow_flag1 = solve1(dataset, lengths, feature_col)
    result2, allow_flag2 = solve2(dataset, feature_col)

    result = np.column_stack((result1, result2))
    allow_flag = combine_allow_flag(allow_flag1, allow_flag2)

    return result, allow_flag
