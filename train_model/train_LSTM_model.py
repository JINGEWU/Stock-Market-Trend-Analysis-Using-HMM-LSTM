from dataset_code.process_on_raw_data import form_raw_dataset
import pickle
from dataset_code import HMM_duoyinzi, HMM_hangqing
from dataset_code.pred_proba_GMM import pred_proba_GMM
from dataset_code.combine import combine
from dataset_code.pred_proba_XGB import pred_proba_XGB
from train_model.LSTM import self_LSTM


def train_LSTM_model():
    # train the LSTM model based on state_proba formed by GMM_HMM or XGB_HMM

    # generate the dataset

    feature_col_hangqing = ['preClosePrice', 'openPrice', 'closePrice', 'turnoverVol', 'highestPrice', 'lowestPrice']
    score, feature_name = HMM_duoyinzi.load_duoyinzi_single_score()
    feature_col_duoyinzi = HMM_duoyinzi.type_filter(score, feature_name, 0.1)
    feature_col = feature_col_hangqing
    _ = [[feature_col.append(j) for j in i] for i in feature_col_duoyinzi]
    dataset, label, lengths, col_nan_record = form_raw_dataset(feature_col, 5)

    # 1 by GMM_HMM
    # 1.1 hangqing
    solved_dataset1, allow_flag1 = HMM_hangqing.solve_on_raw_data(dataset, lengths, feature_col)
    model = pickle.load(open('C:/Users/Administrator/Desktop/HMM_program/save/hangqing_GMM_HMM_model.pkl', 'rb'))
    pred_proba1 = pred_proba_GMM(model, solved_dataset1, allow_flag1, lengths)

    # 1.2 duoyinzi
    pred_proba2 = []
    allow_flag2 = []
    model = pickle.load(open('C:/Users/Administrator/Desktop/HMM_program/save/duoyinzi_GMM_HMM_model.pkl', 'rb'))
    for i in range(len(feature_col_duoyinzi)):
        temp_solved_dataset, temp_allow_flag = HMM_duoyinzi.solve_on_raw_data(dataset, lengths, feature_col, feature_col_duoyinzi[i])
        temp_model = model[i]
        temp_pred_proba = pred_proba_GMM(temp_model, temp_solved_dataset, temp_allow_flag, lengths)
        pred_proba2.append(temp_pred_proba)
        allow_flag2.append(temp_allow_flag)

    # 1.3 combine two type state_proba
    final_X, final_y, final_lengths = combine(pred_proba1, pred_proba2, allow_flag1, allow_flag2, label, lengths)

    # 1.4 train LSTM model
    self_LSTM(final_X, final_y, final_lengths, 'GMM_HMM_LSTM')

    # 2 by XGB_HMM
    # 2.1 hangqing
    solved_dataset1, allow_flag1 = HMM_hangqing.solve_on_raw_data(dataset, lengths, feature_col)
    temp = pickle.load(open('C:/Users/Administrator/Desktop/HMM_program/save/hangqing_XGB_HMM_model.pkl', 'rb'))
    A, model, pi = temp[0], temp[1], temp[2]
    pred_proba1 = pred_proba_XGB(A, model, pi, solved_dataset1, allow_flag1, lengths)

    # 2.2 duoyinzi
    pred_proba2 = []
    allow_flag2 = []
    model = pickle.load(open('C:/Users/Administrator/Desktop/HMM_program/save/duoyinzi_XGB_HMM_model.pkl', 'rb'))
    for i in range(len(feature_col_duoyinzi)):
        temp_solved_dataset, temp_allow_flag = HMM_duoyinzi.solve_on_raw_data(dataset, lengths, feature_col, feature_col_duoyinzi[i])
        temp_A, temp_model, temp_pi = model[i][0], model[i][1], model[i][2]
        temp_pred_proba = pred_proba_XGB(temp_A, temp_model, temp_pi, temp_solved_dataset, temp_allow_flag, lengths)
        pred_proba2.append(temp_pred_proba)
        allow_flag2.append(temp_allow_flag)

    # 2.3 combine two type state_proba
    final_X, final_y, final_lengths = combine(pred_proba1, pred_proba2, allow_flag1, allow_flag2, label, lengths)

    # 2.4 train LSTM model
    self_LSTM(final_X, final_y, final_lengths, 'XGB_HMM_LSTM')
