# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

# import seaborn as sns
# sns.set()

import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['font.serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.pylab import date2num
# from mpl_finance import candlestick_ochl
# from mpl_finance import candlestick_ohlc
from mplfinance.original_flavor import candlestick_ochl
from mplfinance.original_flavor import candlestick_ohlc
# from .mpfOld import candlestick_ohlc
# from .mpfOld import candlestick_ochl
from matplotlib.gridspec import GridSpec


def plot_Candle(data, N=50,
                args_ma=[5, 10, 20, 30, 60],
                args_boll=[15, 2],
                plot_below='volume', args_ma_below=[3, 5, 10],
                figsize=(15, 10), fig_save_path=None, title=None, fontsize=20,
                width=0.5, alpha=0.95, grid=True):
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
        cols_must = ['time', 'open', 'high', 'low', 'close'] + [plot_below]
        if not all([x in data.columns for x in cols_must]):
            raise ValueError(
                'data必须包含time, open, high, low, close及plot_below指定的列！')
    
    cols = ['time', 'open', 'high', 'low', 'close']
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
    
    
def plot_Candle2(data, N=50,
                 args_ma=[5, 10, 20, 30, 60],
                 args_boll=[15, 2],
                 plot_below='volume', args_ma_below=[3, 5, 10], 
                 figsize=(15, 10), fig_save_path=None,
                 title=None, fontsize=20,
                 width=0.7, alpha=0.95, grid=True):
    '''
    绘制K线图（蜡烛图）
    
    Args:
        data: 绘图数据，必须有'time'、'open'、'high'、'low'、'close'五列，
              以及plot_below指定的列名
        N: 用于绘图的数据量大小（从后往前取N条记录画图）
        args_ma: 绘制指定移动均线（MA）列表，None时不绘制
        args_boll: 绘制布林带参数[lag, width]
            注：args_ma和args_boll必须至少有一个为None
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
        cols_must = ['time', 'open', 'high', 'low', 'close'] + [plot_below]
        if not all([x in data.columns for x in cols_must]):
            raise ValueError(
                'data必须包含time, open, high, low, close及plot_below指定的列！')
    
    cols = ['time', 'open', 'high', 'low', 'close']
    cols = cols + [plot_below] if plot_below else cols
    data = data.reindex(columns=cols).iloc[-N:, :]
    
    # K线图数据准备
    data['time'] = pd.to_datetime(data['time'])
    data['time'] = data['time'].apply(lambda x: date2num(x))
    data_K = data[['time', 'open', 'high', 'low', 'close']].values

    # 坐标准备
    if plot_below:
        fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=figsize)
    else:
        fig, ax1 = plt.subplots(1, figsize=figsize)
    
    # K线图
    candlestick_ohlc(ax1, data_K, colorup='r', colordown='g',
                                 width=width, alpha=alpha)
    ax1.xaxis_date()
    ax1.grid(grid)
    ax1.set_ylabel('价格(or点位)', {'fontsize': fontsize})
    
    # 标题绘制在K线图顶部
    if title:
        ax1.set_title(title, {'fontsize': fontsize})
        
    if args_ma and args_boll:
        raise ValueError('均线和布林带只能绘制一种！')
    
    # 均线图
    if args_ma:
        args_ma = [x for x in args_ma if x < N]
        if len(args_ma) > 0:
            for m in args_ma:
                data['MA'+str(m)] = data['close'].rolling(m).mean()
                ax1.plot(data['time'], data['MA'+str(m)], label='MA'+str(m))
            ax1.legend(loc=0)
            
    # 布林带
    if args_boll:
        data['boll_mid'] = data['close'].rolling(args_boll[0]).mean()
        data['boll_std'] = data['close'].rolling(args_boll[0]).std()
        data['boll_up'] = data['boll_mid'] + args_boll[1] * data['boll_std']
        data['boll_low'] = data['boll_mid'] - args_boll[1] * data['boll_std']
        ax1.plot(data['time'], data['boll_mid'], '-k')
        ax1.plot(data['time'], data['boll_low'], '-r')
        ax1.plot(data['time'], data['boll_up'], '-g')
        ax1.legend(loc=0)
            
    # 底部图
    if plot_below:
        data['up'] = data.apply(lambda x: 1 if x['close'] >= x['open'] else 0,
                                axis=1)
        ax2.bar(data.query('up == 1')['time'], 
                data.query('up == 1')[plot_below], color='r',
                width=width+0.1, alpha=alpha)
        ax2.bar(data.query('up == 0')['time'], 
                data.query('up == 0')[plot_below], color='g',
                width=width+0.1, alpha=alpha)
        ax2.set_ylabel(plot_below, {'fontsize': fontsize})        
        ax2.grid(grid)
        
        # 底部均线图
        if args_ma_below:
            args_ma_below = [x for x in args_ma_below if x < N]
            if len(args_ma_below) > 0:
                for m in args_ma_below:
                    data['MA'+str(m)] = data[plot_below].rolling(m).mean()
                    ax2.plot(data['time'], data['MA'+str(m)],
                             label='MA'+str(m))
                ax2.legend(loc=0)
        
    plt.xlabel('时间', {'fontsize': fontsize})
    
    plt.tight_layout()
    
    if fig_save_path:
        plt.savefig(fig_save_path)
        
    plt.show()


if __name__ == '__main__':
    from utils_hoo import load_csv
    
    daily_50etf_pre_fq_path = '../test/510050_daily_pre_fq.csv'
    data = load_csv(daily_50etf_pre_fq_path)
    data.rename(columns={'date': 'time'}, inplace=True)
    
    N = 100
#    args_ma = None
    args_ma = [5, 10, 20, 30, 50]
    args_boll = None
#    args_boll = [15, 2]
#    plot_below = None
    plot_below = 'volume'
#    args_ma_below = None
    args_ma_below = [3, 5, 10]
    figsize = (11, 10)
#    fig_save_path = None
    fig_save_path = './plot_test/Candle_test.png'
    title = '50ETF'
    fontsize = 20
    width = 0.5
    alpha = 0.95
    grid = True
    
    plot_Candle(data, N=N, args_ma=args_ma, args_boll=args_boll,
                plot_below=plot_below, args_ma_below=args_ma_below,
                figsize=figsize, fig_save_path=fig_save_path, title=title,
                fontsize=fontsize, width=width, alpha=alpha, grid=grid)
    plot_Candle2(data, N=N, args_ma=args_ma, args_boll=args_boll,
                plot_below=plot_below, args_ma_below=args_ma_below,
                figsize=figsize, fig_save_path=fig_save_path, title=title,
                fontsize=fontsize, width=width, alpha=alpha, grid=grid)
