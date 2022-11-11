"""
    train the LSTM model
"""


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.layers.advanced_activations import LeakyReLU
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelBinarizer
from public_tool.bagging_balance_weight import bagging_balance_weight
from keras.layers import BatchNormalization
from keras.optimizers import SGD
from public_tool.form_index import form_index
from public_tool.random_cut import random_cut
from public_tool.form_accuracy import form_accuracy
from keras.models import load_model
import pickle


def form_LSTM_dataset(X, y, lengths, T):
    # 形成LSTM专用的数据类型
    # input:
    #     X
    #     y
    #     lengths
    #     T: 往前使用的数据的步长
    # output:
    #     result_X: array (n_sample, T, n_features)
    #     result_y: array (n_sample, )

    result_X = []
    result_y = []
    for i in range(len(lengths)):
        begin_index, end_index = form_index(lengths, i)
        now_X = X[begin_index:end_index]
        now_y = y[begin_index:end_index]

        for j in range(len(now_y) - T):
            result_X.append(now_X[j:j + T])
            result_y.append(now_y[j + T])

    result_X = np.array(result_X)

    return result_X, result_y


def self_LSTM(X, y, lengths, file_name):
    """

    :param X:
    :param y:
    :param lengths:
    :param file_name: save the result [mms, model] to the indicated file_name, for example: 'XGB_HMM_LSTM'
    """

    # normalization
    mms = MinMaxScaler()
    X = mms.fit_transform(X)

    T = 10
    X_LSTM, y_LSTM = form_LSTM_dataset(X, y, lengths, T)

    # one hot
    lb = LabelBinarizer()
    y_LSTM = lb.fit_transform(y_LSTM)

    # random cut
    X_train, y_train, X_valid, y_valid = random_cut(X_LSTM, y_LSTM, 5)
    X_train, y_train = bagging_balance_weight(X_train, y_train)

    # creat and fit model
    model = Sequential()

    model.add(Dropout(0.2))
    model.add(LSTM(40))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.02))

    model.add(Dropout(0.2))
    model.add(Dense(30, kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.02))

    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # es = EarlyStopping(monitor='val_loss', min_delta=3e-4, patience=10, verbose=1, mode='auto')
    model.fit(X_train,
              y_train,
              batch_size=int(X_train.shape[0] / 5),
              epochs=10000,
              verbose=2,
              validation_split=0.0,
              validation_data=(X_valid, y_valid),
              shuffle=True,
              class_weight=None,
              sample_weight=None,
              initial_epoch=0,
              callbacks=[])

    print('train    ', form_accuracy(X_train, y_train, model))
    print('valid   ', form_accuracy(X_valid, y_valid, model))

    mms_file_name = '/content/drive/MyDrive/Projects/Database/Models/' + file_name + '_mms.pkl'
    pickle.dump(mms, open(mms_file_name, 'wb'))
    model_file_name = '/content/drive/MyDrive/Projects/Database/Models/' + file_name + '_model.h5'
    model.save(model_file_name)
