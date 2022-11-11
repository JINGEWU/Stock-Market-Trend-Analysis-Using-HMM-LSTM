import numpy as np
import pandas as pd
import pickle

def replace_price_0_to_nan(df):
    # For the price data, if it is 0, assign the value to nan and leave it to the subsequent interpolation fillNAN module for processing

    col_list = ['preClose','Open', 'High', 'Low', 'Close']
    for i in col_list:
        temp = np.array(df[i].values)
        temp[temp == 0] = np.nan
        df[i] = temp
    return df


def replace_vol_0_to_1(df):
    # For the data whose vol is 0, when the log is processed later, it will cause inf, so replace it with 1

    col_list = ['Volume']
    for i in col_list:
        temp = np.array(df[i].values)
        temp[temp == 0] = 1
        df[i] = temp
    return df


def form_label(df, threshold_type='ratio', threshold=0.05, T=5):
    # input:
    #     df: dataframe
    #     threshold_type: 'ratio' or 'specific'
    #     threshold: value
    #     T: length of triple barries
    # output:
    #     label: array, (df.shape[0], )
    #     The output result is 0, -1, 1, -2, where -2 means the length is not enough
    
    df.sort_values(['tradingDate'], inplace=True, ascending=True)
    
    close_price_array = np.array(df['close'].values)
    label_array = np.zeros(len(close_price_array))-2
    for i in range(len(close_price_array)):
        if len(close_price_array)-i-1 < T:
            continue
        else:
            now_close_price = close_price_array[i]
            
            if threshold_type == 'ratio':
                temp_threshold = now_close_price*threshold
            else:
                temp_threshold = threshold
            
            flag = 0
            for j in range(T):
                if close_price_array[i+j+1]-now_close_price > temp_threshold:
                    label_array[i] = 1
                    flag = 1
                    break
                elif close_price_array[i+j+1]-now_close_price < -temp_threshold:
                    label_array[i] = -1
                    flag = 1
                    break
            if flag == 0:
                label_array[i] = 0
                
    return label_array


def array_isnan(array):
    # input:
    #   Array type, one-dimensional and two-dimensional, the data in it is int, str, float, nan.
    # output:
    #   Array type, the size is the same as the previous data, it is True and False

    result = np.zeros(array.shape)
    if len(array.shape) == 1:
        for i in range(array.shape[0]):
            data = array[i]
            if isinstance(data, str):
                result[i] = False
            else:
                result[i] = np.isnan(data)
    if len(array.shape) == 2:
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                data = array[i, j]
                if isinstance(data, str):
                    result[i] = False
                else:
                    result[i] = np.isnan(data)
                    
    return result



def fill_na(array, N_error=5):
    """
     input:
         array: col victor
         N_error: how many consecutive nans indicate error
     output:
         1. 'error', str, means that there are 5 consecutive nans
         2, array, representing the result after interpolation
    """

    error_flag = 0
    count = 0
    for i in range(len(array)):
        if not type(array[i]) == str:
            if np.isnan(array[i]):
                count += 1
            else:
                count = 0
        else:
            count = 0
        if count >= N_error:
            error_flag = 1
            break
    
    if error_flag == 0:
        temp = pd.DataFrame(array)
        
        na_index = temp.loc[temp.isnull().iloc[:, 0]].index - temp.index[0]
        
        if len(na_index) > 0:
            
            y = temp.dropna().iloc[:, 0]
            x = temp.dropna().index.values - temp.index[0]
            t = interpolate.splrep(x, y, s=0)

            y_filled = interpolate.splev(na_index, t)
    
            temp.iloc[na_index, 0] = y_filled
        
            if 0 in na_index:
                temp.iloc[0, 0] = sum(temp.iloc[1:6, 0])/5
            if temp.shape[0]-1 in na_index:
                temp.iloc[temp.shape[0]-1, 0] = sum(temp.iloc[-6:-1, 0])/5
            
        return np.array(temp.iloc[:, 0].values)
    else:
        return 'error'


def tran_nan(array):
    # Unify the nan of the array as np.nan, and change the object of the array to float type
    result = np.zeros(array.shape)
    if len(array.shape) == 1:
        for i in range(len(array)):
            if not type(array[i]) == str:
                if np.isnan(array[i]):
                    result[i] = np.nan
                else: 
                    result[i] = array[i]
            else: 
                result[i] = array[i]
    else:
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                if not type(array[i, j]) == str:
                    if np.isnan(array[i, j]):
                        result[i, j] = np.nan
                    else:
                        result[i, j] = array[i, j]
                else:
                    result[i, j] = array[i, j]
                        
    return result


def form_raw_dataset(feature_col, label_length, intID_select_list=None, verbose=True):
    # By default only import data from a specific field
    # According to the required feature_col (list type), form X, label, lengths (array type)
    # Where X is already processed, does not contain nan, after interpolation, the singular value of 0 becomes 0.1
    # input:
    #   feature_col: Column name of the data to process
    #   label_length: time length of triple bars
    #   intID_select_list: list, select the int code of the stock that generated the sample
    #   verbose: whether to output print information
    # output:
    #   X, label, lengths, col_nan_record (record how many nans there are in each column)
    
        
    now_df = pickle.load(open('/content/drive/MyDrive/Projects/Database/dataset/market_factor/MWG_stock.pkl', 'rb'))
    
    now_df = replace_price_0_to_nan(now_df)
    now_df = replace_vol_0_to_1(now_df)
 
    
    now_label = form_label(now_df, threshold_type='ratio', threshold=0.05, T=label_length)
    now_X = tran_nan(now_df[feature_col].values)
    
    drop_flag = 0
    lengths = []
    
    for k in range(now_X.shape[1]):
        temp = fill_na(now_X[:, k])
        if type(temp) == str:
            drop_flag = 1
            break
        else:
            now_X[:, k] = temp
            
    if drop_flag == 0:
        if init_flag == 1:
            X = now_X
            label = now_label
            lengths = [len(label)]
            init_flag = 0
        else:
            X = np.row_stack((X, now_X))
            label = np.hstack((label, now_label))
            lengths.append(len(now_label))
                
    
    return X, label, lengths