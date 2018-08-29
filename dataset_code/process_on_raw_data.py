"""
    由规定的feature_col，得到满足条件的链的数据
"""


import pandas as pd
import numpy as np
import os
import pickle
from scipy import interpolate


def form_file_path_by_intID(intID):
    # input:
    #     int类型，股票代码的int
    # output:
    #     str类型，文件的路径，可以直接使用

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
    #     df, dataframe
    # output:
    #     有些时候merge完，如果merge的两个df有同样的col_name，那么新的df会有col_name_x, col_name_y
    #     那么这时候就把这两个列名称删掉，只保留一个，并且保留的那个的列名称变为col_name
    #     返回处理后的结果

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
    # 针对价格的数据，如果是0的话，赋值为nan，留给后面的插值fillna模块处理

    col_list = ['preClosePrice', 'actPreClosePrice', 'openPrice', 'highestPrice', 'lowestPrice', 'closePrice']
    for i in col_list:
        temp = np.array(df[i].values)
        temp[temp == 0] = np.nan
        df[i] = temp
    return df


def replace_vol_0_to_1(df):
    # 针对vol是0的数据，对于后面取log处理的时候，会导致inf，所以取代为1

    col_list = ['turnoverVol', 'turnoverValue', 'dealAmount']
    for i in col_list:
        temp = np.array(df[i].values)
        temp[temp == 0] = 1
        df[i] = temp
    return df


def fenge_by_isOpen(df, N=50):
    # input:
    #     df, 正常格式下的df
    #     N, 保证分割出来的df的长度要大于N
    # output:
    #     list类型，里面装了分割后的df

    # 将持续5天及以上没有开盘，那么就分割
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
    #     输出结果为0，-1，1，-2，其中-2表示长度不够
    
    df.sort_values(['tradeDate'], inplace=True, ascending=True)
    
    close_price_array = np.array(df['closePrice'].values)
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
    #     数组类型，一维的二维的都行，里面的数据是int，str，float，nan都行
    # output:
    #     数组类型，大小跟之前的数据一样，是True和False

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
    #     返回列表类型
    #     遍历所有的pkl中间文件，得到每个列名称的nan的数量统计，得到nan比例超过threshold的列名称

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
    # 返回feature的列名称，列表类型
    
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
        N_error: 连续多少个nan表示error
    output:
        1、'error', str， 代表有存在连续5个nan
        2、array，代表插值过后的结果
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
    # 将array的nan统一为np.nan，而且将数组的object变为float类型

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
    # 默认只是导入特定领域的数据
    # 根据所需要的feature_col(list类型)，形成X，label，lengths（array类型）
    # 其中X是已经处理过，不包含nan，经过插值，0的奇异值变为0.1
    # input:
    #     feature_col: 要处理的数据的列名称
    #     label_length: triple barries 的时间长度
    #     intID_select_list: list, 选择生成样本的股票的int代码
    #     verbose: 是否要输出打印信息
    # output:
    #     X, label, lengths, col_nan_record(记录每一列有多少个nan的情况)

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
