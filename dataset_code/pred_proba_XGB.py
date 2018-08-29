import numpy as np
from public_tool.form_index import form_index
from XGB_HMM.form_B_matrix_by_XGB import form_B_matrix_by_XGB
from XGB_HMM.predict import self_pred


def pred_proba_XGB(A, model, pi, O, allow_flag, lengths):
    # 对dataset形成pred_proba，注意这里的dataset是solve_on_raw_data后的结果，即附带allow_flag的数据
    # output:
    #     pred_proba：数组类型

    n_states = len(pi)
    pred_proba = np.zeros((O.shape[0], n_states))

    for i in range(len(lengths)):
        begin_index, end_index = form_index(lengths, i)

        now_O = O[begin_index:end_index, :]
        now_allow_flag = allow_flag[begin_index:end_index]
        now_pred_proba = np.zeros((now_O.shape[0], n_states))

        now_allow_B = form_B_matrix_by_XGB(model, now_O[now_allow_flag == 1], pi)
        _, now_allow_pred_proba, _ = self_pred(now_allow_B, [now_allow_B.shape[0]], A, pi)

        now_pred_proba[now_allow_flag == 1] = now_allow_pred_proba
        pred_proba[begin_index:end_index] = now_pred_proba

    return pred_proba
