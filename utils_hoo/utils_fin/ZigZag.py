# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from utils_hoo import load_csv


import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['font.serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.pylab import date2num
from mplfinance.original_flavor import candlestick_ochl
from matplotlib.gridspec import GridSpec


def plot_Candle_zz(data, N=50, zzcol='zigzag', zz_high='high', zz_low='low',
                   args_ma=[5, 10, 20, 30, 60],
                   args_boll=[15, 2],
                   plot_below='volume', args_ma_below=[3, 5, 10],
                   figsize=(15, 10), fig_save_path=None, title=None,
                   fontsize=20, width=0.5, alpha=0.95, grid=True):
    '''
    绘制K线图（蜡烛图）
    
    Args:
        data: 绘图数据，必须有'time'、'open'、'high'、'low'、'close'五列，
              以及plot_below指定的列名
        N: 用于绘图的数据量大小（从后往前取N条记录画图）
        args_ma: 绘制指定移动均线（MA）列表，None时不绘制
        args_boll: 绘制布林带参数[lag, width]
            注：args_ma和args_boll必须有一个为None
        plot_below: 在K线底部绘制柱状图所用的列名，None时不绘制
        args_ma_below: 底部图均线（MA）列表，None时不绘制
        fig_size: 图像大小
        fig_save_path: 图像保存路径，None时不保存
        title: 图像标题
        fontsize: 图例字体大小
        width: 控制蜡烛宽度
        alpha: 控制颜色透明度
        grid: 设置是否显示网格
    '''
    
    if plot_below:
        cols_must = ['time', 'open', 'high', 'low', 'close', zzcol] + \
                                                                  [plot_below]
        if not all([x in data.columns for x in cols_must]):
            raise ValueError(
        'data必须包含time, open, high, low, close, zigzag及plot_below指定的列！')
    
    cols = ['time', 'open', 'high', 'low', 'close', zzcol]
    cols = cols + [plot_below] if plot_below else cols
    data = data.reindex(columns=cols).iloc[-N:, :]

    data['date_bkup'] = data['time'].copy()
    data['time'] = pd.to_datetime(data['time']).map(date2num)
    data['timeI'] = np.arange(0, data.shape[0])

    date_tickers = data['date_bkup'].values
    
    def format_date(x, pos):
        if x < 0 or x > len(date_tickers)-1:
            return ''
        return date_tickers[int(x)]

    # 坐标准备
    plt.figure(figsize=figsize)
    if plot_below:
        gs = GridSpec(3, 1)
        ax1 = plt.subplot(gs[:2, :])
        ax2 = plt.subplot(gs[2, :])
    else:
        gs = GridSpec(1, 1)
        ax1 = plt.subplot(gs[:, :])

    # 绘制K线图
    data_K = data[['timeI', 'open', 'close', 'high', 'low']].values
    candlestick_ochl(ax=ax1, quotes=data_K, width=width, 
                                 colorup='r', colordown='g', alpha=alpha)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
    ax1.grid(grid)
    ax1.set_ylabel('价格(or点位)', fontsize=fontsize)
    
    # 标题绘制在K线图顶部
    if title is not None:
        ax1.set_title(title, fontsize=fontsize)
        
    if args_ma and args_boll:
        raise ValueError('均线和布林带只能绘制一种！')
        
    # zigzag
    data['col_zz1'] = data[zzcol].apply(lambda x: 1 if x > 0 else 0)
    data['col_zz-1'] = data[zzcol].apply(lambda x: 1 if x < 0 else 0)
    data['col_zz'] = data['col_zz1'] * data[zz_high] + \
                                               data['col_zz-1'] * data[zz_low]
    df_zz = data[data['col_zz'] != 0][['timeI', 'col_zz']]
    ax1.plot(df_zz['timeI'], df_zz['col_zz'], '-b')

    # 均线图
    if args_ma:
        args_ma = [x for x in args_ma if x < N]
        if len(args_ma) > 0:
            for m in args_ma:
                data['MA'+str(m)] = data['close'].rolling(m).mean()
                ax1.plot(data['timeI'], data['MA'+str(m)], label='MA'+str(m))
            ax1.legend(loc=0)
            
    # 布林带
    if args_boll:
        data['boll_mid'] = data['close'].rolling(args_boll[0]).mean()
        data['boll_std'] = data['close'].rolling(args_boll[0]).std()
        data['boll_up'] = data['boll_mid'] + args_boll[1] * data['boll_std']
        data['boll_low'] = data['boll_mid'] - args_boll[1] * data['boll_std']
        ax1.plot(data['timeI'], data['boll_mid'], '-k')
        ax1.plot(data['timeI'], data['boll_low'], '-r')
        ax1.plot(data['timeI'], data['boll_up'], '-g')
        ax1.legend(loc=0)
    
    # 底部图
    if plot_below:
        ax2.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
        data['up'] = data.apply(lambda x: 1 if x['close'] >= x['open'] else 0,
                                axis=1)
        ax2.bar(data.query('up == 1')['timeI'], 
                data.query('up == 1')[plot_below], color='r',
                width=width+0.1, alpha=alpha)
        ax2.bar(data.query('up == 0')['timeI'], 
                data.query('up == 0')[plot_below], color='g',
                width=width+0.1, alpha=alpha)
        ax2.set_ylabel(plot_below, fontsize=fontsize)        
        ax2.grid(grid)
        
        # 底部均线图
        if args_ma_below:
            args_ma_below = [x for x in args_ma_below if x < N]
            if len(args_ma_below) > 0:
                for m in args_ma_below:
                    data['MA'+str(m)] = data[plot_below].rolling(m).mean()
                    ax2.plot(data['timeI'], data['MA'+str(m)],
                             label='MA'+str(m))
                ax2.legend(loc=0)
    
    plt.xlabel('时间', fontsize=fontsize)
    
    plt.tight_layout()
    
    if fig_save_path:
        plt.savefig(fig_save_path)
        
    plt.show()


def ZigZag(data, high_col='high', low_col='low', up_pct=1/100, down_pct=1/100):
    '''
    ZigZag转折点
    
    Parameters
    ----------
    data: pd.DataFrame，需包含[high_col, low_col]列
    high_col/low_col: 确定zigzag高/低点的数据列名
    up_pct/down_pct: 确定zigzag高/低转折点的幅度
    
    Returns
    -------
    data['zigzag']: zigzag标签序列，其中1/-1表示确定的高/低点，
                    0.5/-0.5表示未达到偏离幅度而不能确定的高低点
    '''
    
    def confirm_high(k):
        '''从前一个低点位置k开始确定下一个高点位置'''
        v0, pct_high, pct_high_low = df.loc[df.index[k], low_col], 0.0, 0.0
        Cmax, Cmax_idx = -np.inf, k+1
        k += 2
        while k < df.shape[0] and \
                              (pct_high_low > -down_pct or pct_high < up_pct):
            if df.loc[df.index[k-1], high_col] > Cmax:
                Cmax = df.loc[df.index[k-1], high_col]
                Cmax_idx = k-1
            pct_high = Cmax / v0 - 1
            
            pct_high_low = min(pct_high_low,
                                   df.loc[df.index[k], low_col] / Cmax - 1)
            
            k += 1
            
        if k == df.shape[0]:
            if df.loc[df.index[k-1], high_col] > Cmax:
                Cmax = df.loc[df.index[k-1], high_col]
                Cmax_idx = k-1
                pct_high = Cmax / v0 - 1
                pct_high_low = 0.0
                
        return Cmax_idx, pct_high >= up_pct, pct_high_low <= -down_pct
    
    def confirm_low(k):
        '''从前一个高点位置k开始确定下一个低点位置'''
        v0, pct_low, pct_low_high = df.loc[df.index[k], high_col], 0.0, 0.0
        Cmin, Cmin_idx = np.inf, k+1
        k += 2
        while k < df.shape[0] and \
                              (pct_low_high < up_pct or pct_low > -down_pct):
            if df.loc[df.index[k-1], low_col] < Cmin:
                Cmin = df.loc[df.index[k-1], low_col]
                Cmin_idx = k-1
            pct_low = Cmin / v0 - 1
            
            pct_low_high = max(pct_low_high,
                                   df.loc[df.index[k], high_col] / Cmin - 1)
            
            k += 1  
            
        if k == df.shape[0]:
            if df.loc[df.index[k-1], low_col] < Cmin:
                Cmin = df.loc[df.index[k-1], low_col]
                Cmin_idx = k-1
                pct_low = Cmin / v0 - 1
                pct_low_high = 0.0
                
        return Cmin_idx, pct_low <= -down_pct, pct_low_high >= up_pct
    
    # 若data中已有zigzag列，先检查找出最后一个能转折点已经能确定的位置，从此位置开始算
    if 'zigzag' in data.columns:
        cols = list(set([high_col, low_col])) + ['zigzag']
        df = data[cols].copy()
        k = df.shape[0] - 1
        while k > 0 and df.loc[df.index[k], 'zigzag'] in [0, 0.5, -0.5]:
            k -= 1
        ktype = df.loc[df.index[k], 'zigzag']
    # 若data中没有zigzag列或已有zigzag列不能确定有效的转折点，则从头开始算
    if 'zigzag' not in data.columns or ktype in [0, 0.5, -0.5]:
        cols = list(set([high_col, low_col]))
        df = data[cols].copy()
        df['zigzag'] = 0
        # 确定开始时的高/低点标签
        k1, OK_high, OK_high_low = confirm_high(0)
        OK1 = OK_high and OK_high_low
        k_1, OK_low, OK_low_high = confirm_low(0)
        OK_1 = OK_low and OK_low_high
        if not OK1 and not OK_1:
            return df['zigzag']
        elif OK1 and not OK_1:
            df.loc[df.index[0], 'zigzag'] = -0.5
            df.loc[df.index[k1], 'zigzag'] = 1
            k = k1
            ktype = 1
        elif not OK1 and OK_1:
            df.loc[df.index[0], 'zigzag'] = 0.5
            df.loc[df.index[k_1], 'zigzag'] = -1
            k = k_1
            ktype = -1
        elif k1 < k_1:
            df.loc[df.index[0], 'zigzag'] = -0.5
            df.loc[df.index[k1], 'zigzag'] = 1
            k = k1
            ktype = 1
        else:
            df.loc[df.index[0], 'zigzag'] = 0.5
            df.loc[df.index[k_1], 'zigzag'] = -1
            k = k_1
            ktype = -1
    
    while k < df.shape[0]:
        func_confirm = confirm_high if ktype == -1 else confirm_low
        k, OK_mid, OK_right = func_confirm(k)
        if OK_mid and OK_right:
            df.loc[df.index[k], 'zigzag'] = -ktype
            ktype = -ktype
        elif OK_mid:
            df.loc[df.index[k], 'zigzag'] = -ktype * 0.5
            break
            
    return df['zigzag']


if __name__ == '__main__':
    # zigzig测试--------------------------------------------------------------
    fpath = '../test/510050_daily_pre_fq.csv'
    his_data = load_csv(fpath)
    his_data.rename(columns={'date': 'time'}, inplace=True)
    his_data.set_index('time', drop=False, inplace=True)
        
    # N = his_data.shape[0]
    N = 100
    col = 'close'
    data = his_data.iloc[-N:-1, :].copy()
    
    high_col, low_col, up_pct, down_pct = 'high', 'low', 3/100, 3/100
    data['zigzag'] = ZigZag(data, high_col, low_col, up_pct, down_pct)
    plot_Candle_zz(data, N=data.shape[0], zz_high=high_col, zz_low=low_col,
                   args_ma=None, args_boll=None, plot_below=None, grid=False,
                   figsize=(12, 7))
    
    
    fpath = '../test/zigzag_test.csv'
    data = load_csv(fpath)
    dates = list(data['date'].unique())
    data = data[data['date'] == dates[0]].copy()
    plot_Candle_zz(data, N=data.shape[0], zzcol='label', args_ma=None,
                    args_boll=None, plot_below=None, figsize=(12, 7))
    
    high_col, low_col, up_pct, down_pct = 'high', 'low', 0.35/100, 0.35/100
    data['zigzag'] = ZigZag(data, high_col, low_col, up_pct, down_pct)
    plot_Candle_zz(data, N=data.shape[0], args_ma=None, args_boll=None,
                    plot_below=None, figsize=(12, 7))
    
    