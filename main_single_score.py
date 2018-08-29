"""
    得到一个因子一个分数的记录列表
"""

import pickle
import numpy as np
from dataset_code.process_on_raw_data import form_raw_dataset, df_col_quchong
from dataset_code.HMM_duoyinzi import solve2, form_model_dataset, form_model
from public_tool.evaluate_plot import evaluate_plot
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    
    temp = pickle.load(open('save/classified by id/000001.XSHE.pkl', 'rb'))
    temp = df_col_quchong(temp)
    temp = [i for i in temp.columns]
    feature_list = temp[temp.index('AccountsPayablesTDays'):]
    score_record = np.zeros(len(feature_list))

    for i in range(len(feature_list)):

        now_feature = [feature_list[i]]

        dataset, label, lengths, col_nan_record = form_raw_dataset(now_feature, label_length=3, verbose=False)

        if len(label) == 0:
            print('skip ' + now_feature[0])
            continue

        solved_dataset, allow_flag = solve2(dataset, now_feature, now_feature)

        train_X, train_label, train_lengths = form_model_dataset(solved_dataset, label, allow_flag, lengths)

        model = form_model(train_X, train_lengths, 3, 'diag', 1000, verbose=False)

        score = evaluate_plot(model, train_X, train_label, train_lengths)
        score_record[i] = score

        print('all:%s, now:%s, ' % (len(feature_list), i + 1) + now_feature[0] + ': score:%s' % score)

        pickle.dump([score_record, feature_list], open('save/duoyinzi_solve2_score.pkl', 'wb'))

    pickle.dump([score_record, feature_list], open('save/duoyinzi_solve2_score.pkl', 'wb'))
