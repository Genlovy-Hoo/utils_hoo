# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy import stats
from utils_hoo.utils_datsci.utils_stats import Fit_norm
from utils_hoo.utils_datsci.utils_stats import Fit_lognorm
from utils_hoo.utils_datsci.utils_stats import Fit_weibull

import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['font.serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt


def plot_HistDist(series, bins=None, width=None, clor='grey',
                  density=False, kde_styl=None, dists={'norm': ('-r', None)}, 
                  ylabel_left=None, ylabel_right=None, con_points=10000, 
                  xticks_gap=1, figsize=(12, 8), title=None, fontsize=15,
                  fig_save_path=None):
    
    '''
    绘制直方图，及分布拟合曲线
    
    Parameters
    ----------
    series: 待绘图序列，pd.Series
    bins: None或list或整数，指定区间划分方式
    width: 直方图宽度设置
    clor: 直方图颜色
    density: True或False，直方图坐标是否转为改概率密度
    kde_styl: None或(lnstyl, clor, label)，若为None则不绘制核密度图kde，
        若为后者，lnsty设置线型，可为None或str；clor设置颜色，可为None或str；label设置
        图例内容，可为None或str或False，为None时默认'kde'，为False时不设置图例
    dists: 概率密度分布绘图信息列表，None（不绘制概率分布拟合曲线）或如下格式: 
        {disttype: (lnstyl, label)}或{disttype: lnstyl}，其中disttype指定概率密度
        函数类型；lnstyl设置线型，可为str或None，为None时自动设置线型；第一种格式中label
        设置图例内容，可为str或None或False，为None时图例设置为disttype，为False时不设置
        图例，第二种格式图例默认设置为disttype
        注：disttype目前可选'norm'、'lognorm'、'weibull'
    ylabel_left, ylabel_right: 左右y轴标签内容
    con_points: 概率密度曲线拟合时在x轴采样点个数
    xticks_gap: 设置每xticks_gap个直方图矩形框显示一个x轴刻度
    '''
    
    # 序列名和索引名
    if series.name is None:
        series.name = 'series'
        
    # 直方图bins
    bins = int(np.sqrt(len(series)) + 1) if bins is None else bins
    N = int(bins) if not isinstance(bins, list) else len(bins)
    Smax, Smin = series.max(), series.min()
    if not isinstance(bins, list):
        gap = (Smax-Smin) / N
        bins = [Smin + k*gap for k in range(0, N)]
        
    # 坐标轴准备
    _, ax1 = plt.subplots(figsize=figsize)
    if kde_styl is not None or dists is not None:
        ax2 = ax1.twinx()
    lns = [] # 存放legends信息
    
    # 直方图绘制
    if width is not None:
        ln = ax1.hist(series, bins=bins, color=clor, width=width, align='mid',
                      edgecolor='black', density=density, label=series.name)
    else:
        ln = ax1.hist(series, bins=bins, color=clor, align='mid',
                      edgecolor='black', density=density, label=series.name)
        # hist返回对象中第一个值为区间计数，第二个值为区间边界，第三个值才是图形handle
        lns.append(ln[2])
        
    # 左轴标签
    if ylabel_left is None:
        ylabel_left = '密度函数' if density else '频数'
    ax1.set_ylabel(ylabel_left, fontsize=fontsize)
    
    # 核密度kde绘制
    if kde_styl is not None:
        lnstyl, clor_kde, lbl_str = kde_styl        
        lbl_str = 'kde' if lbl_str is None else lbl_str
        
        if not lbl_str and str(lbl_str)[0] != '0':
            series.plot(kind='kde', ax=ax2, linestyle=lnstyl, color=clor_kde)
        else:
            ln = series.plot(kind='kde', ax=ax2, linestyle=lnstyl,
                             color=clor_kde, label=lbl_str)
            lns.append(ln.lines) # 取出ln中的lines句柄
            
    # 指定概率分布拟合
    if dists is not None:
        funcs_fit = {'norm': Fit_norm, 'lognorm': Fit_lognorm,
                     'weibull': Fit_weibull}
        
        x = np.arange(Smin, Smax, (Smax-Smin)/con_points)
        
        for dist, styl in dists.items():
            y = funcs_fit[dist](series, x)
            
            if styl is None:
                lnstyl, lbl_str = '-', dist
            else:
                if isinstance(styl, str):
                    lnstyl, lbl_str = styl, dist
                else:
                    lnstyl, lbl_str = styl
            lnstyl = '-' if lnstyl is None else lnstyl
            lbl_str = dist if lbl_str is None else lbl_str
            
            if lbl_str is False:
                ax2.plot(x, y, lnstyl)
            else:    
                ln = ax2.plot(x, y, lnstyl, label=lbl_str)
                lns.append(ln)
            
    # 右轴标签
    if kde_styl is not None or dists is not None:
        if ylabel_right == False:
            ax2.set_ylabel(None)
            ax2.set_yticks([])
        else:
            ylabel_right = 'P' if ylabel_right is None else ylabel_right
            ax2.set_ylabel(ylabel_right, fontsize=fontsize)
        
    # 合并legends
    lnsAdd = [lns[0][0]] # hist返回的图形handle列表中，只有第一个有legend
    for ln in lns[1:]:
        lnsAdd = lnsAdd + [ln[0]]
    labs = [l.get_label() for l in lnsAdd]
    ax1.legend(lnsAdd, labs, loc=0, fontsize=fontsize)
        
    xpos = bins[:-1:xticks_gap]
    xticks = [round(x,2) if not isinstance(x, int) else x for x in xpos]
    plt.xticks(xpos, xticks)
    
    plt.tight_layout()
    
    if title:
        plt.title(title, fontsize=fontsize)
        
    if fig_save_path:
        plt.savefig(fig_save_path)
    
    plt.show()
    
    
if __name__ == '__main__':
    # series = pd.read_excel('./plot_test/percent.xlsx')['percent']
    # series = pd.read_csv('./plot_test/series1.csv')['series']
    # series = pd.read_csv('./plot_test/series2.csv')['series']
    
    # series = pd.Series(np.random.normal(5, 3, (1000,)))
    
    # 注：numpy中lognormal分布mean和sigma参数的意义为
    # np.log(series)服从参数为mean和sigma正态分布
    # series = pd.Series(np.random.lognormal(mean=1, sigma=1, size=(1000,)))
    # 注: stats中lognorm分布参数和numpy中参数关系：
    # 若设置loc = 0，则有s = sigma，scale = e ^ mean
    # series = pd.Series(stats.lognorm(s=1, loc=0, scale=np.exp(1)).rvs(1000,))
    
    # 注：stats中weibull_min分布参数和np.random.weibull分布参数关系：
    # 若设置loc=0，则有c = a，scale = 1
    # series = pd.Series(np.random.weibull(a=2, size=(1000,)))
    series = pd.Series(stats.weibull_min(c=2, loc=0, scale=1).rvs(1000,))
    
    # bins = list(range(-25, 55, 5))
    bins = 15
    # bins = None
    # width = 1
    width = None
    clor = 'yellow'
    density = False
    kde_styl = None
    # kde_styl = ('-', 'b', None)
    # dists = {'norm': ('-r', 'N')}
    dists = {'norm': ('-r', None), 'lognorm': ('-g', None),
             'weibull': ('-k', 'weibull')}
    ylabel_left = None
    ylabel_right = None
    con_points = 10000
    xticks_gap = 2
    figsize = (10, 8)
    title = '直方图（拟合）'
    fontsize = 20
    fig_save_path = './plot_test/HistDist_test.png'
    # fig_save_path = None
    
    plot_HistDist(series, bins=bins, width=width, clor=clor,
                  density=density, kde_styl=kde_styl, dists=dists, 
                  ylabel_left=ylabel_left, ylabel_right=ylabel_right,
                  con_points=con_points, xticks_gap=xticks_gap,
                  figsize=figsize, title=title, fontsize=fontsize,
                  fig_save_path=fig_save_path)
    