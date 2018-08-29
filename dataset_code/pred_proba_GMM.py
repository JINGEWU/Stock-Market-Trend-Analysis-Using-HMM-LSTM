import numpy as np
from public_tool.form_index import form_index


def pred_proba_GMM(model, O, allow_flag, lengths):
    # 对dataset形成pred_proba，注意这里的dataset是solve_on_raw_data后的结果，即附带allow_flag的数据
    # output:
    #     pred_proba：数组类型

    pred_proba = np.zeros((O.shape[0], model.n_components))

    for i in range(len(lengths)):
        begin_index, end_index = form_index(lengths, i)

        now_O = O[begin_index:end_index, :]
        now_allow_flag = allow_flag[begin_index:end_index]
        now_pred_proba = np.zeros((now_O.shape[0], model.n_components))

        now_pred_proba[now_allow_flag == 1] = model.predict_proba(now_O[now_allow_flag == 1])

        pred_proba[begin_index:end_index] = now_pred_proba

    return pred_proba
