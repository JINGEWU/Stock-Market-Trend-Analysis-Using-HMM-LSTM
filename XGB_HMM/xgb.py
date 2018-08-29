import xgboost as xgb
import numpy as np


def self_xgb(X, gamma, n_states):

    params = {'objective': 'multi:softprob',
              'learning_rate': 0.01,
              'colsample_bytree': 0.886,
              'min_child_weight': 3,
              'max_depth': 10,
              'subsample': 0.886,
              'reg_alpha': 1.5,  # L1的正则系数
              'reg_lambda': 0.5,  # L2的正则系数
              'gamma': 0.5,  # 分裂阈值
              'n_jobs': -1,
              'eval_metric': 'mlogloss',
              'scale_pos_weight': 1,
              'random_state': 201806,
              'missing': None,
              'silent': 1,
              'max_delta_step': 0,
              'num_class': n_states}

    y = np.array([np.argmax(i) for i in gamma])
    temp = np.array([np.max(i) for i in gamma])
    y = y[temp >= 0.9]
    X = X[temp >= 0.9]

    sample_weight = temp[temp >= 0.9]

    d_train = xgb.DMatrix(X, y, weight=sample_weight)

    model = xgb.train(params, d_train, num_boost_round=1000, verbose_eval=True)

    pred = np.array([np.argmax(i) for i in model.predict(d_train)])
    print(sum(pred == y)/len(y))

    return model
