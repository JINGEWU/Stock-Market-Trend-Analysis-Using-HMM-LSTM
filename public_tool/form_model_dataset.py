from public_tool.form_index import form_index
import numpy as np


def form_model_dataset(dataset, label, allow_flag, lengths):
    # # 根据HMM1_solve的结果，形成后面用于训练HMM模型的数据类型
    # input:
    #     dataset, HMM1_solve
    #     label, HMM1_solve
    #     allow_flag, HMM1_solve
    #     lengths
    # output:
    #     result_dataset
    #     result_lengths

    result_dataset = np.zeros((0, dataset.shape[1]))
    result_label = np.zeros(0)
    result_lengths = []

    for i in range(len(lengths)):
        begin_index, end_index = form_index(lengths, i)

        now_dataset = dataset[begin_index:end_index, :]
        now_allow_flag = allow_flag[begin_index:end_index]
        now_label = label[begin_index:end_index]

        result_dataset = np.row_stack((result_dataset, now_dataset[now_allow_flag == 1]))
        result_label = np.hstack((result_label, now_label[now_allow_flag == 1]))
        result_lengths.append(sum(now_allow_flag == 1))

    return result_dataset, result_label, result_lengths
