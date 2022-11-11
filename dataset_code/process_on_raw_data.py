"""
     From the specified feature_col, get the data of the chain that meets the conditions
"""


import pandas as pd
import numpy as np
import os
import pickle
from scipy import interpolate


def form_file_path_by_intID(intID):
    # input:
    #   int type, int of stock symbol
    # output:
    #   str type, the path of the file, can be used directly

    temp = os.listdir('C:/Users/Administrator/Desktop/HMM_program/save/classified by id')
    intID_list = [int(i[:-9]) for i in temp]
    strID_list = [i[:-9] for i in temp]
    strExchange_list = [i[-9:-4] for i in temp]
    
    if not (intID in intID_list):
        return 'None'
    else:
        index = np.where(np.array(intID_list) == intID)[0][0]
        strID = strID_list[index]
        Exchange = strExchange_list[index]
        file_path = 'C:/Users/Administrator/Desktop/HMM_program/save/classified by id/' + strID + Exchange + '.pkl'
        return file_path


def df_col_quchong(df):
    # input:
    #   df, dataframe
    # output:
    #   Sometimes the merge is finished, if the two dfs of the merge have the same col_name, then the new df will have col_name_x, col_name_y
    #   Then delete these two column names at this time, keep only one, and the column name of the reserved one becomes col_name
    #   return the processed result

    feature_col = [i for i in df.columns]
    warm_record = []
    for i in feature_col:
        if i[-2:] == '_x' or i[-2:] == '_y':
            warm_record.append(i)
    
    true_col = []
    error_col = []
    
    while len(warm_record) != 0:
        now_warm = warm_record[0]
        now_warm_part = now_warm[0:-2]
        now_warm_col = [now_warm]
        flag = 1
        for i in range(len(warm_record)-1):
            i = i+1
            if now_warm_part == warm_record[i][0:-2]:
                now_warm_col.append(warm_record[i])
                if flag == 1:
                    true_col.append(now_warm_part)
                    flag = 0
        if len(now_warm_col) > 1:
            error_col.append(now_warm_col)
        for i in now_warm_col:
            warm_record.remove(i)
    
    for i in range(len(true_col)):
        now_true_col = true_col[i]
        now_error_col = error_col[i]
        df = df.rename(columns={now_error_col[0]: now_true_col})
        now_error_col.remove(now_error_col[0])
        df.drop(now_error_col, axis=1, inplace=True)
    
    return df.copy()


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


def fenge_by_isOpen(df, N=50):
    # input:
    #   df, df in normal format
    #   N, ensure that the length of the divided df is greater than N
    # output:
    #   list type, which contains the divided df

    # Will last 5 days or more without opening, then split
    df_record = []
    df.sort_values(['tradeDate'], inplace=True, ascending=True)
    
    isopen = np.array(df['isOpen'].values)
    pre_index = 0
    df_flag = 1
    for end_index in range(len(isopen)):
        if df_flag == 1:
            if sum(isopen[end_index+1:end_index+6]) == 0:
                temp = df.loc[pre_index:end_index]
                if temp.shape[0] > N:
                    df_record.append(temp)
                df_flag = 0
            else:
                continue
        else:
            if isopen[end_index] == 1:
                pre_index = end_index
                df_flag = 1
            else:
                continue
    
    return df_record


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
    
    close_price_array = np.array(df['Close'].values)
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
    

def col_with_high_ratio_nan(threshold):
    # output:
    #   Return list type
    #   Traverse all pkl intermediate files, get the statistics of the number of nans for each column name, 
    #   and get the column names whose nan ratio exceeds the threshold

    if os.path.exists('save/col_na_ratio.pkl'):
        temp = pickle.load(open('save/col_na_ratio.pkl', 'rb'))
        count = temp[0]
        np_count = temp[1]
        col_list = temp[2]
    else:
        file_list = os.listdir('save/classified by id')

        init_flag = 1
        for i in range(len(file_list)):
            df = pickle.load(open('save/classified by id/'+file_list[i], 'rb'))
            result = array_isnan(df.values)
            if init_flag == 1:
                col_list = [i for i in df.columns]
                np_count = np.zeros(len(col_list))
                count = np.zeros(len(col_list))
                init_flag = 0
            
            count += df.shape[0]
            np_count += np.sum(result, axis=0)
            
            print('all:%s, now:%s' % (len(file_list), i+1))
        
        pickle.dump([count, np_count, col_list], open('save/col_na_ratio.pkl', 'wb'))
    
    ratio = np_count/count
    del_col = []
    for i in range(len(ratio)):
        if ratio[i] > threshold:
            del_col.append(col_list[i])
            
    return del_col
        

def form_feature_name(threshold=0.1):
    # Return the column name of feature, list type
    
    temp = os.listdir('save/classified by id')
    temp = pickle.load(open('save/classified by id/'+temp[0], 'rb'))
    temp = df_col_quchong(temp)
    
    feature_col = [i for i in temp.columns]
    feature_col.remove('secID')
    feature_col.remove('ticker')
    feature_col.remove('secShortName')
    feature_col.remove('exchangeCD')
    feature_col.remove('tradeDate')
    feature_col.remove('actPreClosePrice')
    
    temp = col_with_high_ratio_nan(threshold)
    for i in temp:
        if i in feature_col:
            feature_col.remove(i)
    
    feature_col.append('ratio_openPrice')
    feature_col.append('ratio_highestPrice')
    feature_col.append('ratio_lowestPrice')
    feature_col.append('ratio_closePrice')
    
    return feature_col


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

    # temp = pd.read_table('data/dianzixinxi.txt').secID.values
    # intID_select_list = [i for i in temp]

    if intID_select_list is None:
        temp = pd.read_table('C:/Users/Administrator/Desktop/HMM_program/data/dianzixinxi.txt')
        intID_select_list = [i for i in temp['secID']]

    init_flag = 1
    select = []
    
    col_nan_record = np.zeros(len(feature_col))
    
    for i in range(len(intID_select_list)):
        now_intID = intID_select_list[i]
        now_file_path = form_file_path_by_intID(now_intID)
        if now_file_path == 'None':
            continue
        
        now_df = pickle.load(open(now_file_path, 'rb'))
        
        now_df = df_col_quchong(now_df)
        now_df = replace_price_0_to_nan(now_df)
        now_df = replace_vol_0_to_1(now_df)
        
        now_df_record = fenge_by_isOpen(now_df)
        
        for j in range(len(now_df_record)):
            now_df1 = now_df_record[j].copy()
            
            now_label = form_label(now_df1, threshold_type='ratio', threshold=0.05, T=label_length)
            now_X = tran_nan(now_df1[feature_col].values)
            
            drop_flag = 0
            for k in range(now_X.shape[1]):
                temp = fill_na(now_X[:, k])
                if type(temp) == str:
                    drop_flag = 1
                    col_nan_record[k] += 1
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
                select.append(now_df1.head(1)['secShortName'].values[0])
                
        if verbose:
            if init_flag == 1:
                print('all:%s, finished:%s' % (len(intID_select_list), i+1))
            else:
                print('all:%s, finished:%s, len_X:%s, num_chain:%s' % (len(intID_select_list), i+1, X.shape[0], len(select)))
    
    print(col_nan_record)
    print(feature_col)
    
    if init_flag == 1:
        return None
    
    return X, label, lengths, col_nan_record
