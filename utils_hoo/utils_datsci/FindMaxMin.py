# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from utils_hoo import load_csv
from utils_hoo.utils_plot.plot_Common import plot_MaxMins
from utils_hoo.utils_general import con_count

    
def FindMaxMin(series, Tmin=2):
    '''
    todo: 解决series.index有重复值时报错情况：先重置index最后再还原
    
    寻找序列series的极值点
    
    Parameters
    ----------
    series: pd.Series，待寻找极值点的序列
    Tmin: 设置极大极小值之间至少需要间隔Tmin个点（相当于最小半周期）
    
    Returns
    -------
    label: 'label'列，其值中1表示极大值点，-1表示极小值点，0表示普通点
    '''
    
    if len(series) < 2:
        raise ValueError('输入series长度不能小于2！')
        
    if not isinstance(series, pd.core.series.Series):
        series = pd.Series(series)
    
    # 序列名和索引名
    series.name = 'series'
    if series.index.name is None:
        series.index.name = 'idx'
    
    df = pd.DataFrame(series)
    col = df.columns[0]
    df['dif'] = series.diff()
    
    # 极大值点
    df['dif_big'] = (df['dif'] > 0).astype(int)
    df['big_rep'] = df['dif_big'].cumsum()
    df['big_rep'] = df[df['dif_big'] == 0]['big_rep'].diff()
    df['big_rep'] = df['big_rep'].shift(-1).fillna(0)
    df.loc[df.index[0], 'big_rep'] = 1 if df['dif'].iloc[1] < 0 else 0
    df.loc[df.index[-1], 'big_rep'] = 1 if df['dif'].iloc[-1] > 0 else 0
    
    # 极小值点
    df['dif_sml'] = (df['dif'] < 0).astype(int)
    df['sml_rep'] = df['dif_sml'].cumsum()
    df['sml_rep'] = df[df['dif_sml'] == 0]['sml_rep'].diff()
    df['sml_rep'] = df['sml_rep'].shift(-1).fillna(0)
    df.loc[df.index[0], 'sml_rep'] = 1 if df['dif'].iloc[1] > 0 else 0
    df.loc[df.index[-1], 'sml_rep'] = 1 if df['dif'].iloc[-1] < 0 else 0
    
    df['label'] = df[['big_rep', 'sml_rep']].apply( lambda x:
        1 if x['big_rep'] > 0 else (-1 if x['sml_rep'] > 0 else 0), axis=1)
    
    df.reset_index(inplace=True)
    
    # plot_MaxMins(df, col, 'label', title='Ori')
        
    # 保证极大极小值必须是间隔的，不能连续出现极大值或连续出现极小值
    # 注：如果序列中存在相邻的值相等，则按上面方法可能可能出现连续的极大/小值点
    k = 0
    while k < df.shape[0]:
        if df.loc[k, 'label'] == 0:
            k += 1
        elif df.loc[k, 'label'] == -1:
            k1 = k
            idxs = [] # 连续极小值点索引列表
            while k1 < df.shape[0] and df.loc[k1, 'label'] != 1:
                if df.loc[k1, 'label'] == -1:
                    idxs.append(k1)
                k1 += 1
            if len(idxs) > 1:
                for n in idxs:
                    # 保留最小的极小值点（不可能出现值相等的连续极大/小值点）
                    if df.loc[n, col] == df.loc[idxs, col].min():
                        df.loc[n, 'label'] == -1
                    else:
                        df.loc[n, 'label'] = 0
            k = k1
        else:
            k1 = k
            idxs = [] # 连续极大值点索引列表
            while k1 < df.shape[0] and df.loc[k1, 'label'] != -1:
                if df.loc[k1, 'label'] == 1:
                    idxs.append(k1)
                k1 += 1
            if len(idxs) > 1:
                for n in idxs:
                    # 保留最大的极大值点（不可能出现值相等的连续极大/小值点）
                    if df.loc[n, col] == df.loc[idxs, col].max():
                        df.loc[n, 'label'] == 1
                    else:
                        df.loc[n, 'label'] = 0
            k = k1
    
    # Tmin应大于等于1
    if Tmin is not None and Tmin < 1:
        Tmin = None
    
    if Tmin:
        def del_Tmin(df):
            '''
            删除不满足最小半周期的极值点对（由一个极大一个极小两个极值点组成），删除条件：
                1：间隔小于Tmin
                2：删除后不影响趋势拐点
            注：df中数据的排序依据为df.index
            '''
            
            k2 = 0
            while k2 < df.shape[0]:
                if df.loc[k2, 'label'] == 0:
                    k2 += 1
                else:
                    k1 = k2-1
                    while k1 > -1 and df.loc[k1, 'label'] == 0:
                        k1 -= 1
                        
                    k3 = k2+1
                    while k3 < df.shape[0] and df.loc[k3, 'label'] == 0:
                        k3 += 1 
                        
                    k4 = k3 +1
                    while k4 < df.shape[0] and df.loc[k4, 'label'] == 0:
                        k4 += 1
                       
                    # 删除条件1
                    if k3-k2 < Tmin+1 and k4 < df.shape[0] and k1 > -1:
                        if df.loc[k2, 'label'] == 1:
                            # 删除条件2
                            if df.loc[k2, col] <= df.loc[k4, col] and \
                                           df.loc[k3, col] >= df.loc[k1, col]:
                                df.loc[[k2, k3], 'label'] = 0
                                
                        else:
                            # 删除条件2
                            if df.loc[k2, col] >= df.loc[k4, col] and \
                                           df.loc[k3, col] <= df.loc[k1, col]:
                                df.loc[[k2, k3], 'label'] = 0
                            
                    # 开头部分特殊处理
                    elif k3-k2 < Tmin+1 and k4 < df.shape[0] and k1 < 0:
                        if df.loc[k2, 'label'] == 1 and \
                                        df.loc[k2, col] < df.loc[k4, col]:
                            df.loc[k2, 'label'] = 0
                        if df.loc[k2, 'label'] == -1 and \
                                        df.loc[k2, col] > df.loc[k4, col]:
                            df.loc[k2, 'label'] = 0
                            
                    k2 = k3
                
            return df
        
        df = del_Tmin(df)
        # plot_MaxMins(df, col, 'label', title='1st Tmin check')
        
        df.index = range(df.shape[0]-1, -1, -1)
        df = del_Tmin(df)
        
        def check_Tmin(df, Tmin):
            Fcond = lambda x: True if x == 0 else False
            df['tmp'] = con_count(df['label'], Fcond).shift(1)
            df['tmp'] = abs(df['tmp'] * df['label'])
            df.loc[df.index[0], 'tmp'] = 0
            tmp = list(df[df['label'] != 0]['tmp'])
            df.drop('tmp', axis=1, inplace=True)
            if len(tmp) <= 3:
                return True, tmp
            else:
                tmp = tmp[1:]
                if all([x >= Tmin for x in tmp]):
                    return True, tmp
                else:
                    return False, tmp
        TminOK, tmp = check_Tmin(df, Tmin)
        tmp_new = []
        # plot_MaxMins(df, col, 'label', title='Tmin check: '+str(TminOK))
        # 注：特殊情况下不可能满足任何两个极大极小值对之间的间隔都大于Tmin
        while not TminOK and not tmp == tmp_new:
            TminOK, tmp = check_Tmin(df, Tmin)
            df.index = range(df.shape[0])
            df = del_Tmin(df)
            df.index = range(df.shape[0]-1, -1, -1)
            df = del_Tmin(df)
            TminOK, tmp_new = check_Tmin(df, Tmin)
            # plot_MaxMins(df, col, 'label', title='Tmin check: '+str(TminOK))
        
    df.set_index(series.index.name, inplace=True)
        
    return df['label']


def check_peaks(df, col, col_label, max_lbl=1, min_lbl=-1):
    '''
    检查df中col_label指定列的极值点排列是否正确
    要求df须包含指定的两列，其中：
        col_label指定列保存极值点，
        # 其中1表示极大值，-1表示极小值，0表示普通点
        其中max_lbl表示极大值，min_lbl表示极小值，其余为普通点
        （max_lbl和min_lbl须为整数）
        另一列为序列数值列
    '''
        
    tmp = df[[col, col_label]].reset_index()
    # df_part = tmp[tmp[col_label].isin([1, -1])]
    df_part = tmp[tmp[col_label].isin([max_lbl, min_lbl])]
    
    if df_part.shape[0] == 0:
        return False, '没有发现极值点，检查输入参数！'
    if df_part.shape[0] == 1:
        if df_part[col].iloc[0] in [df[col].max(), df[col].min()]:
            return True, '只发现1个极值点！'
        else:
            return False, '只发现一个极值点且不是最大或最小值！'
    if df_part.shape[0] == 2:
        vMax = df_part[df_part[col_label] == max_lbl][col].iloc[0]
        vMin = df_part[df_part[col_label] == min_lbl][col].iloc[0]
        if vMax <= vMin:
            return False, '只发现两个极值点且极大值小于等于极小值！'
        if vMax != df[col].max():
            return False, '只发现两个极值点且极大值不是最大值！'
        if vMin != df[col].min():
            return False, '只发现两个极值点且极小值不是最小值！'
        return True, '只发现两个极值点！'
    
    # 不能出现连续的极大/极小值点
    label_diff = list(df_part[col_label].diff().unique())
    if 0 in label_diff:
        return False, '存在连续极大/极小值点！'
    
    # 极大/小值点必须大/小于极小/大值点
    for k in range(1, df_part.shape[0]-1):
        # if df_part[col_label].iloc[k] == 1:
        if df_part[col_label].iloc[k] == max_lbl:
            if df_part[col].iloc[k] <= df_part[col].iloc[k-1] or \
                            df_part[col].iloc[k] <= df_part[col].iloc[k+1]:
                return False, ('极大值点小于等于极小值点！',
                               df.index[df_part.index[k]])
        else:
            if df_part[col].iloc[k] >= df_part[col].iloc[k-1] or \
                            df_part[col].iloc[k] >= df_part[col].iloc[k+1]:
                return False, ('极小值点大于等于极大值点！',
                               df.index[df_part.index[k]])
    
    # 极大极小值点必须是闭区间内的最大最小值
    for k in range(0, df_part.shape[0]-1):
        idx1 = df_part.index[k]
        idx2 = df_part.index[k+1]
        # if tmp.loc[idx1, col_label] == 1:
        if tmp.loc[idx1, col_label] == max_lbl:
            if tmp.loc[idx1, col] != tmp.loc[idx1:idx2, col].max() or \
                    tmp.loc[idx2, col] != tmp.loc[idx1:idx2, col].min():
                return False, ('极大极小值不是闭区间内的最大最小值！',
                               [df.index[idx1], df.index[idx2]])
        else:
            if tmp.loc[idx1, col] != tmp.loc[idx1:idx2, col].min() or \
                    tmp.loc[idx2, col] != tmp.loc[idx1:idx2, col].max():
                return False, ('极大极小值不是闭区间内的最大最小值！',
                               [df.index[idx1], df.index[idx2]])
            
    return True, None


if __name__ == '__main__':
    # # 二次曲线叠加正弦余弦-------------------------------------------------------
    # N = 200
    # t = np.linspace(0, 1, N)
    # s = 6*t*t + np.cos(10*2*np.pi*t*t) + np.sin(6*2*np.pi*t)
    # df = pd.DataFrame(s, columns=['test'])
    
    # # Tmin = None
    # Tmin = 10
    # df['label'] = FindMaxMin(df['test'], Tmin=Tmin)
   
    # plot_MaxMins(df, 'test', 'label', 
    #              title='寻找极大极小值test：Tmin='+str(Tmin))
   
    # OK, e = check_peaks(df, df.columns[0], df.columns[1])
    # if OK:
    #     print('极值点排列正确！')
    # else:
    #     print('极值点排列错误:', e)
   
    
    # 50ETF日线行情------------------------------------------------------------
    fpath = '../test/510050_daily_pre_fq.csv'
    his_data = load_csv(fpath)
    his_data.set_index('date', drop=False, inplace=True)
        
    # N = his_data.shape[0]
    N = 500
    col = 'close'
    df = his_data.iloc[-N:, :].copy()
    
    # Tmin = None
    Tmin = 3
    df['label'] = FindMaxMin(df[col], Tmin=Tmin)
    
    plot_MaxMins(df.iloc[:, :], col, 'label', figsize=(12, 7))
    
    OK, e = check_peaks(df, col, 'label')
    if OK:
        print('极值点排列正确！')
    else:
        print('极值点排列错误:', e)
        
        
    # # 50ETF分钟行情------------------------------------------------------------
    # fpath = '../../../HooFin/data/Archive/index_minute/000016.csv'
    # his_data = load_csv(fpath)
    # his_data['time'] = his_data['time'].apply(lambda x: x[11:])
    # his_data.set_index('time', drop=False, inplace=True)
       
    # df = his_data.iloc[-240*4:-240*3, :].copy()
    # N = df.shape[0]
    # # N = 1000
    # col = 'close'
    # df = df.iloc[-N:, :]

    # # Tmin = None
    # Tmin = 30
    # df['label'] = FindMaxMin(df[col], Tmin=Tmin)

    # plot_MaxMins(df, col, 'label')

    # OK, e = check_peaks(df, col, 'label')
    # if OK:
    #     print('极值点排列正确！')
    # else:
    #     print('极值点排列错误:', e)
       