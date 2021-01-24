# -*- coding: utf-8 -*-

import time
import numpy as np
import pandas as pd
from utils_hoo import load_csv
from utils_hoo.utils_general import x_div_y, isnull
from utils_hoo.utils_datsci.utils_stats import AVEDEV
from utils_hoo.utils_plot.plot_Common import plot_Series


def MACD(series, fast=12, slow=26, m=9):
    '''
    注：计算结果与同花顺PC统一版基本能对上，但是跟远航版没对上
    MACD计算
    http://www.360doc.com/content/17/1128/12/50117541_707746936.shtml
    https://baijiahao.baidu.com/s?id=1602850251881203999&wfr=spider&for=pc
    https://www.cnblogs.com/xuruilong100/p/9866338.html
    '''
    
    col = 'series'
    series.name = col
    df = pd.DataFrame(series)
    
    # df['DI'] = (df['high'] + df['close'] + 2*df['low']) / 4
    df['DI'] = df[col].copy()
    
    df['EMA_fast'] = df['DI'].copy()
    df['EMA_slow'] = df['DI'].copy()
    k = 1
    while k < df.shape[0]:
        df.loc[df.index[k], 'EMA_fast'] = \
                       df.loc[df.index[k], 'DI'] * 2 / (fast+1) + \
                       df.loc[df.index[k-1], 'EMA_fast'] * (fast-1) / (fast+1)
        df.loc[df.index[k], 'EMA_slow'] = \
                       df.loc[df.index[k], 'DI'] * 2 / (slow+1) + \
                       df.loc[df.index[k-1], 'EMA_slow'] * (slow-1) / (slow+1)
        k += 1
    df['DIF'] = df['EMA_fast'] - df['EMA_slow']
    df['DEA'] = df['DIF'].copy()
    k = 1
    while k < df.shape[0]:
        df.loc[df.index[k], 'DEA'] = \
                                df.loc[df.index[k], 'DIF'] * 2 / (m+1) + \
                                df.loc[df.index[k-1], 'DEA'] * (m-1) / (m+1)
        k += 1
    df['MACD'] = 2 * (df['DIF'] - df['DEA'])
    
    return df[['MACD', 'DIF', 'DEA']]


def Boll(series, lag=15, width=2, n_dot=3):
    '''
    布林带
    
    Parameters
    ----------
    series: pd.Series, 序列数据
    lag: 历史期数
    width: 计算布林带上下轨时用的标准差倍数（宽度）
    n_dot: 计算结果保留小数位数
        
    Returns
    -------
    df_boll: 布林带，包含原始值(即series)，布林带下轨值(boll_low列)，
             中间值(boll_mid列)，上轨值(boll_up列)，标准差(std列)
    '''
    
    df_boll = pd.DataFrame(series)
    col = df_boll.columns[0]
    df_boll['boll_mid'] = df_boll[col].rolling(lag).mean()
    df_boll['boll_std'] = df_boll[col].rolling(lag).std()
    df_boll['boll_up'] = df_boll['boll_mid'] + width*df_boll['boll_std']
    df_boll['boll_low'] = df_boll['boll_mid'] - width*df_boll['boll_std']
    
    df_boll['boll_mid'] = df_boll['boll_mid'].round(n_dot)
    df_boll['boll_up'] = df_boll['boll_up'].round(n_dot)
    df_boll['boll_low'] = df_boll['boll_low'].round(n_dot)
    
    df_boll = df_boll.reindex(columns=[col, 'boll_low', 'boll_mid', 'boll_up',
                                       'boll_std'])
    
    return df_boll


def CCI(df, col_TP=None, N=14, r=0.015):
    '''
    CCI计算
    
    Parameters
    ----------
    df: pd.DataFrame, 历史行情数据，须包含['high', 'low', 'close']列或col_TP列
    col_TP: 用于计算CCI的价格列，若不指定，则根据['high', 'low', 'close']计算
    
    Returns
    -------
    cci: CCI序列
    
    参考:
        同花顺PC端CCI指标公式
        https://blog.csdn.net/spursping/article/details/104485136
        https://blog.csdn.net/weixin_43055882/article/details/86696954
    '''
    
    if col_TP is not None:
        if col_TP in df.columns:
            df_ = df.reindex(columns=[col_TP])
        else:
            raise ValueError('请检查col_TP列名！')
    else:
        df_ = df.reindex(columns=['high', 'low', 'close'])
        df_[col_TP] = df_[['high', 'low', 'close']].sum(axis=1) / 3
    
    df_['MA'] = df_[col_TP].rolling(N).mean()
    df_['MD'] = df_[col_TP].rolling(N).apply(lambda x: AVEDEV(x))
    
    cci = (df_[col_TP] - df_['MA']) / (df_['MD'] * r)
    
    return cci


def EXPMA(df, N=26, col='close'):
    '''
    EXPMA计算
    
    参考：
        https://blog.csdn.net/zxyhhjs2017/article/details/93499930
        https://blog.csdn.net/ydjcs567/article/details/62249627
        tradingview公式
    '''
    r = 2 / (N + 1)
    df_ = df.reindex(columns=[col])
    df_['expma'] = df_[col]
    for k in range(1, df_.shape[0]):
        x0 = df_.loc[df_.index[k-1], 'expma']
        x = df_.loc[df_.index[k], col]
        df_.loc[df_.index[k], 'expma'] = r * (x - x0) + x0
    return df_['expma']


def lineW_MA(series, N=15):
    '''加权MA，权重呈线性递减'''
    return series.rolling(N).apply(
                    lambda x: np.average(x, weights=list(range(1, len(x)+1))))


def KAMA(series, lag=9, fast=2, slow=30, returnER=False):
    '''
    Kaufman's Adaptive Moving Average (KAMA)
    https://school.stockcharts.com/doku.php?id=technical_indicators:kaufman_s_adaptive_moving_average
    https://www.technicalindicators.net/indicators-technical-analysis/152-kama-kaufman-adaptive-moving-average
    '''
    
    col = 'series'
    series.name = col
    df = pd.DataFrame(series)
    
    df['direction'] = abs(df[col].diff(periods=lag))
    df['diff'] = abs(df[col].diff())
    df['volatility'] = df['diff'].rolling(lag).sum()
    df['ER'] = df[['direction', 'volatility']].apply(lambda x:
                    x_div_y(x['direction'], x['volatility'], v_y0=0), axis=1)
    
    fast_alpha = 2 / (fast + 1)
    slow_alpha = 2 / (slow + 1)
    df['SC'] = (df['ER'] * (fast_alpha - slow_alpha) + slow_alpha) ** 2
    
    df['kama'] = np.nan
    k = lag
    df.loc[df.index[k], 'kama'] = (df[col].iloc[:k+1]).mean()
    k += 1
    while k < df.shape[0]:
        pre_kama = df.loc[df.index[k-1], 'kama']
        sc = df.loc[df.index[k], 'SC']
        price = df.loc[df.index[k], col]
        df.loc[df.index[k], 'kama'] = pre_kama + sc * (price - pre_kama)
        k += 1
        
    if not returnER:
        return df['kama']
    return df['kama'], df['ER']


def DeMarkTD(series, N=9, Lag=4):
    '''
    迪马克TD序列：连续N次出现收盘价高/低于前面第Lag个收盘价就发信号。默认九转序列
    返回df中一列为原始series，一列为label，label取1表示高点信号，-1表示低点信号
    '''
       
    if series.name is None:
        series.name = 'series'
    col = series.name    
    df = pd.DataFrame(series)
    
    # 是否大于前面第Lag个信号
    df[col+'_preLag'] = df[col].shift(Lag)
    df['more_preLag'] = df[col] > df[col+'_preLag']
    df['more_preLag'] = df[[col+'_preLag', 'more_preLag']].apply(lambda x:
                                    0 if isnull(x[col+'_preLag']) else \
                                    (1 if x['more_preLag'] else -1), axis=1)
        
    # 计数
    df['count'] = 0
    k = 0
    while k < df.shape[0]:
        if df.loc[df.index[k], 'more_preLag'] == 0:
            k += 1
        elif df.loc[df.index[k], 'more_preLag'] == 1:
            label = 1
            df.loc[df.index[k], 'count'] = label
            ktmp = k + 1
            while ktmp < df.shape[0] and \
                                df.loc[df.index[ktmp], 'more_preLag'] == 1:
                if label == N:
                    label = 1
                else:
                    label += 1
                df.loc[df.index[ktmp], 'count'] = label
                ktmp += 1
            k = ktmp
        else:
            label = -1
            df.loc[df.index[k], 'count'] = label
            ktmp = k + 1
            while ktmp < df.shape[0] and \
                                df.loc[df.index[ktmp], 'more_preLag'] == -1:
                if label == -N:
                    label = -1
                else:
                    label -= 1
                df.loc[df.index[ktmp], 'count'] = label
                ktmp += 1
            k = ktmp
            
    # 提取有效信号
    df['label'] = df['count'].apply(lambda x: 1 if x == N else \
                                                    (-1 if x == -N else 0))
        
    return df['label']


def T_transform(df, T=5):
    '''
    数据周期转化
    df应包含列['open', 'high', 'low', 'close']
    '''
    raise NotImplementedError
    dfT = df.reindex(columns=['open', 'high', 'low', 'close'])
    
    
    
if __name__ == '__main__':
    strt_tm = time.time()
    
    
    # 50ETF日线行情
    fpath = '../test/510050_daily_pre_fq.csv'
    df = load_csv(fpath)
    df.set_index('date', drop=False, inplace=True)
    # df = df.reindex(columns=['high', 'low', 'close'])

    # CCI --------------------------------------------------------------------
    N = 14
    r = 0.015
    df['cci'] = CCI(df, N=N, r=r)
    
    plot_Series(df.iloc[-200:, :], {'close': ('.-k', False)},
                cols_styl_low_left={'cci': ('.-b', False)},
                xparls_info={'cci': [(100, 'r', '-', 1.3),
                                     (-100, 'r', '-', 1.3)]},
                figsize=(8.5, 7), grids=True)
    
    
    # EXPMA-------------------------------------------------------------------
    N = 5
    df['expma'+str(N)] = EXPMA(df, N, 'close')
    
    # MACD--------------------------------------------------------------------
    macds = MACD(df['close'])
    df = df.merge(macds, how='left', left_index=True, right_index=True)
    
    # 九转序列------------------------------------------------------------------
    N, Lag = 9, 4
    df['dmktd'] = DeMarkTD(df['close'], N, Lag)
    
    
    print(f'used time: {round(time.time()-strt_tm, 6)}s.')
    
    
    
    
    
    
    
