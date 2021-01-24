# -*- coding: utf-8 -*-

import pandas as pd
from utils_hoo.utils_general import con_count, isnull


def get_miss_rate(df, return_type='dict'):
    '''
    计算df中每列的缺失率，return_type可选['dict', 'df']
    '''
    mis_rates = df.isnull().sum() / df.shape[0]
    if return_type == 'dict':
        return mis_rates.to_dict()
    elif return_type == 'df':
        mis_rates = pd.DataFrame(mis_rates).reset_index()
        mis_rates.columns = ['col', 'miss_pct']
        mis_rates.sort_values('miss_pct', ascending=False, inplace=True)
        return mis_rates
    
    
def fillna_ma(series, ma=None, ma_min=2):
    '''
    用移动平均ma填充序列series中的缺失值
    ma设置填充时向前取平均数用的期数，ma_min设置最小期数
    若ma为None，则根据最大连续缺失记录数确定ma期数
    '''
    
    if series.name is None:
        series.name = 'series'        
    col = series.name
    df = pd.DataFrame(series)
    
    if ma is None:
        tmp = con_count(series, lambda x: True if isnull(x) else False)
        ma = 2 * tmp.max()
        ma = max(ma, ma_min*2)
    
    df[col+'_ma'] = df[col].rolling(ma, ma_min).mean()
    
    df[col] = df[[col, col+'_ma']].apply(lambda x:
               x[col] if not isnull(x[col]) else \
               (x[col+'_ma'] if not isnull(x[col+'_ma']) else x[col]), axis=1)
        
    return df[col]
    
    
def fillna_by_mean(df, cols):
    '''
    用列均值替换df（pd.DataFrame）中的无效值，cols指定待替换列
    '''
    for col in cols:
        df[col] = df[col].fillna(df[col].mean())
    return df


def fillna_by_median(df, cols):
    '''
    用列中位数替换df（pd.DataFrame）中的无效值，cols指定待替换列
    '''
    for col in cols:
        df[col] = df[col].fillna(df[col].median())
    return df
    
    
if __name__ == '__main__':
    fpath = '../../../No.60_CART_breakup_predict/data.csv'
    df = pd.read_csv(fpath)
    
    mis_rates = get_miss_rate(df)
    mis_rates = {k: v for k, v in mis_rates.items() if v > 0}
    
    cols = ['act']
    df_ = fillna_by_mean(df, cols)
    df__ = fillna_by_mean(df, cols)
    
    a_ = fillna_ma(df['am'])
    