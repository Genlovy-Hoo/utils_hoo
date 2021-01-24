# -*- coding: utf-8 -*-

# import seaborn as sns
# sns.set()

import numpy as np
import pandas as pd
from utils_hoo.utils_general import get_con_start_end
from utils_hoo.utils_general import simple_logger, isnull

import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['font.serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def plot_HistCumPct():
    '''绘制直方图和累计百分比曲线图'''
    raise NotImplementedError
    
    
def plot_series_with_styls_info(ax, series, styls_info, lnstyl_default='.-',
                                markersize=None, lbl_str_ext='',
                                **kwargs_plot):
    '''
    给定线型设置信息styls_info，在ax上对series绘图，lnstyl_default设置默认线型
    styls_info格式形如：
        ('.-b', 'lbl')或'.-b'
        第一种格式中lbl设置图例（legend），lbl若为None则默认取series列名，若为False，
        则不设置图例，第二种格式只设置线型，legend默认取series列名
    lbl_str_ext设置legend文本后缀（比如双坐标轴情况下在右轴的legend加上'(r)'）
    **kwargs_plot可接收符合ax.plot函数的其它参数
    '''
    
    if styls_info is None:
        lnstyl, lbl_str = lnstyl_default, series.name
    else:
        if isinstance(styls_info, str):
            lnstyl, lbl_str = styls_info, series.name
        else:
            lnstyl, lbl_str = styls_info
    lnstyl = lnstyl_default if lnstyl is None else lnstyl
    lbl_str = series.name if lbl_str is None else lbl_str
    
    if lbl_str is False:
        ax.plot(series, lnstyl, markersize=markersize, **kwargs_plot)
        return None
    else:
        ln = ax.plot(series, lnstyl, label=str(lbl_str)+lbl_str_ext,
                     markersize=markersize, **kwargs_plot)
        return ln


def plot_Series(data, cols_styl_up_left, cols_styl_up_right=None,
                cols_styl_low_left=None, cols_styl_low_right=None,
                cols_to_label_info = {}, xparls_info={},
                yparls_info_up=None, yparls_info_low=None, ylabels=None,
                grids=False, figsize=(12, 9), title=None, nXticks=8,
                fontsize=15, markersize=10, fig_save_path=None, logger=None):
    '''
    todo: markersize分开设置，正常绘制与特殊标注重复绘制问题，
          x轴平行线对应列不一定非要在主图绘制列中选择
          平行线图层绘制在主线下面
          标注图层绘制在线型图层上面（根据输入顺序绘制图层而不是根据坐标轴区域顺序绘制）
    
    pd.DataFrame多列绘图
    
    Parameters
    ----------    
    cols_styl_up_left, cols_styl_up_right, cols_styl_low_left,
    cols_styl_low_right:
        分别指定顶部左轴、顶部右轴、底部左轴和底部右需要绘制的序列及其线型和图例，格式如：
        {'col1': ('.-b', 'lbl1'), 'col2': ...}或{'col1': '.-b', 'col2': ...}
        第一种格式中lbl设置图例（legend），若为None则默认取列名，为False，则不设置图例；
        第二种格式只设置线型，legend默认取列名
    cols_to_label_info: 设置需要特殊标注的列绘图信息，格式形如:
        {col1: [[col_lbl1, (v1, v2, ..), (styl1, styl2, ..), (lbl1, lbl2, ..)],
                [col_lbl2, (v1, v2, ..), ...]],
         col2: ..}，其中col是需要被特殊标注的列，col_lbl为标签列；v指定哪些标签值对应的
        数据用于绘图；styl设置线型；lbl设置图例标签，若为None，则设置为v，若为False，
        则不设置图例标签
    xparls_info: 设置x轴平行线信息，格式形如：
        {col1: [(yval1, clor1, styl1, width1), (yval2, ...)], col2:, ...}，
        其中yval指定平行线y轴位置，clor设置颜色，styl设置线型，width设置线宽
    yparls_info_up, yparls_info_low: 分别设置顶部和底部x轴平行线格式信息，格式形如：
        [(xval1, clor1, styl1, width1), (xval2, clor2, style2, width2), ...]，
        其中xval指定平行线x轴位置，clor设置颜色，styl设置线型，width设置线宽
    ylabels: None或列表，设置四个y轴标签文本内容，若为None则不设置标签文本，
        若为False则既不设置y轴标签文本内容，也不显示y轴刻度
    grids: 设置四个坐标轴网格，若grids=True，则将顶部左轴和底部左轴绘制网格；
        若grids=False，则全部没有网格，grids可设置为列表分别对四个坐标轴设置网格
    '''
    
    df = data.copy()    
    logger = simple_logger() if logger is None else logger
    
    # 网格设置，grids分别设置顶部左边、顶部右边、底部左边、底部右边的网格
    if grids == True:
        grids = [True, False, True, False]
    elif grids == False or grids is None:
        grids = [False, False, False, False]
        
    # y轴标签设置
    if ylabels is None:
        ylabels = [None, None, None, None]        
    
    # 索引列处理
    if df.index.name is None:
        df.index.name = 'idx'        
    idx_name = df.index.name
    if idx_name in df.columns:
        df.drop(idx_name, axis=1, inplace=True)
    df.reset_index(inplace=True)
    
    if cols_styl_low_left is None and cols_styl_low_right is not None:
        logger.warning('当底部图只指定右边坐标轴时，默认绘制在左边坐标轴！')
        cols_styl_low_left, cols_styl_low_right = cols_styl_low_right, None
        
    # 坐标准备
    plt.figure(figsize=figsize)
    if cols_styl_low_left is not None:
        gs = GridSpec(3, 1)
        axUpLeft = plt.subplot(gs[:2, :]) # 顶部为主图，占三分之二高度
        axLowLeft = plt.subplot(gs[2, :])
    else:
        gs = GridSpec(1, 1)
        axUpLeft = plt.subplot(gs[:, :])
        
        
    def get_cols_to_label_info(cols_to_label_info, col):
        '''需要进行特殊点标注的列绘图设置信息获取'''
        
        to_plots = []
        for label_infos in cols_to_label_info[col]:
        
            lbl_col = label_infos[0]
            
            if label_infos[2] is None:
                label_infos = [lbl_col, label_infos[1], [None]*len(label_infos[1]),
                               label_infos[3]]
            
            if label_infos[3] == False:
                label_infos = [lbl_col, label_infos[1], label_infos[2],
                               [False]*len(label_infos[1])]
            elif isnull(label_infos[3]) or \
                                        all([isnull(x) for x in label_infos[3]]):
                label_infos = [lbl_col, label_infos[1], label_infos[2],
                               label_infos[1]]
            
            vals = label_infos[1]
            for k in range(len(vals)):
                series = df[df[lbl_col] == vals[k]][col]
                ln_styl = label_infos[2][k]
                lbl_str = label_infos[3][k]
                to_plots.append([series, (ln_styl, lbl_str)])
            
        return to_plots    

    def get_xparls_info(parls_info, col, clor_default='r',
                        lnstyl_default='--', lnwidth_default=2):
        '''x轴平行线绘图设置信息获取'''
        parls = parls_info[col]
        to_plots = []
        for val, clor, lnstyl, lnwidth in parls:
            clor = clor_default if clor is None else clor
            lnstyl = lnstyl_default if lnstyl is None else lnstyl
            lnwidth = lnwidth_default if lnwidth is None else lnwidth
            to_plots.append([val, clor, lnstyl, lnwidth])
        return to_plots
    
    def get_yparls_info(parls_info, clor_default='r', lnstyl_default='--',
                        lnwidth_default=2):
        '''y轴平行线绘图设置信息获取'''
        to_plots = []
        for val, clor, lnstyl, lnwidth in parls_info:
            clor = clor_default if clor is None else clor
            lnstyl = lnstyl_default if lnstyl is None else lnstyl
            lnwidth = lnwidth_default if lnwidth is None else lnwidth
            val = df[df[idx_name] == val].index[0]
            to_plots.append([val, clor, lnstyl, lnwidth])
        return to_plots
    
    
    # lns存放双坐标legend信息
    # 双坐标轴legend参考：https://www.cnblogs.com/Atanisi/p/8530693.html
    lns = []    
    # 顶部左边坐标轴
    for col, styl in cols_styl_up_left.items():
        ln = plot_series_with_styls_info(axUpLeft, df[col], styl)
        if ln is not None:
            lns.append(ln)            
           
        # 特殊点标注
        if col in cols_to_label_info.keys():
            to_plots = get_cols_to_label_info(cols_to_label_info, col)
            for series, styls_info in to_plots:
                ln = plot_series_with_styls_info(axUpLeft, series, styls_info,
                                    lnstyl_default='ko', markersize=markersize)
                if ln is not None:
                    lns.append(ln)
                
        # x轴平行线
        if col in xparls_info.keys():
            to_plots = get_xparls_info(xparls_info, col)
            for yval, clor, lnstyl, lnwidth in to_plots:
                axUpLeft.axhline(y=yval, c=clor, ls=lnstyl, lw=lnwidth)
                
    # y轴平行线
    if yparls_info_up is not None:
        to_plots = get_yparls_info(yparls_info_up)
        for xval, clor, lnstyl, lnwidth in to_plots:
            axUpLeft.axvline(x=xval, c=clor, ls=lnstyl, lw=lnwidth)
        
    # 顶部左边坐标轴网格       
    axUpLeft.grid(grids[0])
    
    # 标题绘制在顶部图上
    if title is not None:
        axUpLeft.set_title(title, fontsize=fontsize)
        
    # y轴标签文本
    if ylabels[0] is False:
        axUpLeft.set_ylabel(None)
        axUpLeft.set_yticks([])
    else:
        axUpLeft.set_ylabel(ylabels[0], fontsize=fontsize)
        
    # 顶部右边坐标轴
    if cols_styl_up_right is not None:
        axUpRight = axUpLeft.twinx()
        for col, styl in cols_styl_up_right.items():
            ln = plot_series_with_styls_info(axUpRight, df[col], styl,
                                             lbl_str_ext='(r)')
            if ln is not None:
                lns.append(ln)
            
            # 特殊点标注
            if col in cols_to_label_info.keys():
                to_plots = get_cols_to_label_info(cols_to_label_info, col)
                for series, styls_info in to_plots:
                    ln = plot_series_with_styls_info(axUpRight, series,
                                            styls_info, lnstyl_default='ko',
                                    markersize=markersize, lbl_str_ext='(r)')
                    if ln is not None:
                        lns.append(ln)
                    
            # x轴平行线
            if col in xparls_info.keys():
                to_plots = get_xparls_info(xparls_info, col)
                for yval, clor, lnstyl, lnwidth in to_plots:
                    axUpRight.axhline(y=yval, c=clor, ls=lnstyl, lw=lnwidth)
                    
        # 顶部右边坐标轴网格
        axUpRight.grid(grids[1])
        
        # y轴标签文本
        if ylabels[1] is False:
            axUpRight.set_ylabel(None)
            axUpRight.set_yticks([])
        else:
            axUpRight.set_ylabel(ylabels[1], fontsize=fontsize)
        
    # 顶部图legend合并显示
    if len(lns) > 0:
        lnsAdd = lns[0]
        for ln in lns[1:]:
            lnsAdd = lnsAdd + ln
        labs = [l.get_label() for l in lnsAdd]
        axUpLeft.legend(lnsAdd, labs, loc=0, fontsize=fontsize)
    
    
    if cols_styl_low_left is not None:
        # 要绘制底部图时取消顶部图x轴刻度
        # axUpLeft.set_xticks([]) # 这样会导致设置网格线时没有竖线
        axUpLeft.set_xticklabels([]) # 这样不会影响设置网格
        lns = []
        
        # 底部左边坐标轴
        for col, styl in cols_styl_low_left.items():
            ln = plot_series_with_styls_info(axLowLeft, df[col], styl)
            if ln is not None:
                lns.append(ln)
            
            # 特殊点标注
            if col in cols_to_label_info.keys():
                to_plots = get_cols_to_label_info(cols_to_label_info, col)
                for series, styls_info in to_plots:
                    ln = plot_series_with_styls_info(axLowLeft, series,
                        styls_info, lnstyl_default='ko', markersize=markersize)
                    if ln is not None:
                        lns.append(ln)
                    
            # x轴平行线
            if col in xparls_info.keys():
                to_plots = get_xparls_info(xparls_info, col)
                for yval, clor, lnstyl, lnwidth in to_plots:
                    axLowLeft.axhline(y=yval, c=clor, ls=lnstyl, lw=lnwidth)
                    
        # y轴平行线
        if yparls_info_low is not None:
            to_plots = get_yparls_info(yparls_info_low)
            for xval, clor, lnstyl, lnwidth in to_plots:
                axLowLeft.axvline(x=xval, c=clor, ls=lnstyl, lw=lnwidth)
            
        # 底部左边坐标轴网格
        axLowLeft.grid(grids[2])    
        
        # y轴标签文本
        if ylabels[2] is False:
            axLowLeft.set_ylabel(None)
            axLowLeft.set_yticks([])
        else:
            axLowLeft.set_ylabel(ylabels[2], fontsize=fontsize)
        
        # 底部右边坐标轴
        if cols_styl_low_right is not None:
            axLowRight = axLowLeft.twinx()
            for col, styl in cols_styl_low_right.items():
                ln = plot_series_with_styls_info(axLowRight, df[col], styl,
                                                 lbl_str_ext='(r)')
                if ln is not None:
                    lns.append(ln)
                
                # 特殊点标注
                if col in cols_to_label_info.keys():
                    to_plots = get_cols_to_label_info(cols_to_label_info, col)
                    for series, styls_info in to_plots:
                        ln = plot_series_with_styls_info(axLowRight, series,
                                            styls_info, lnstyl_default='ko',
                                    markersize=markersize, lbl_str_ext='(r)')
                        if ln is not None:
                            lns.append(ln)
                       
                # x轴平行线
                if col in xparls_info.keys():
                    to_plots = get_xparls_info(xparls_info, col)
                    for yval, clor, lnstyl, lnwidth in to_plots:
                        axLowRight.axhline(y=yval, c=clor, ls=lnstyl,
                                          lw=lnwidth)
                       
            # 底部右边坐标轴网格
            axLowRight.grid(grids[3]) 
            
            # y轴标签文本
            if ylabels[3] is False:
                axLowRight.set_ylabel(None)
                axLowRight.set_yticks([])
            else:
                axLowRight.set_ylabel(ylabels[3], fontsize=fontsize)
                
        # 底部图legend合并显示
        if len(lns) > 0:
            lnsAdd = lns[0]
            for ln in lns[1:]:
                lnsAdd = lnsAdd + ln
            labs = [l.get_label() for l in lnsAdd]
            axLowLeft.legend(lnsAdd, labs, loc=0, fontsize=fontsize)
        
    
    # x轴刻度
    n = df.shape[0]
    xpos = [int(x*n/nXticks) for x in range(0, nXticks)] + [n-1]
    plt.xticks(xpos, [df.loc[x, idx_name] for x in xpos])
    
    plt.tight_layout()
        
    # 保存图片
    if fig_save_path:
        plt.savefig(fig_save_path)
        
    plt.show()
    
    
def plot_Series_conlabel(data, conlabel_info, del_repeat_lbl=True, **kwargs):
    '''
    连续标注绘图
    conlabel_info格式：
    {col: [[lbl_col, (v1, ...), (styl1, ...), (lbl1, ...)]]}
    '''
    
    df_ = data.copy()
    df_['_tmp_idx_'] = range(0, df_.shape[0])
    
    kwargs_new = kwargs.copy()
    if 'cols_to_label_info' in kwargs_new.keys():
        cols_to_label_info = kwargs_new['cols_to_label_info']
    else:
        cols_to_label_info = {}
    
    def deal_exist_lbl_col(col, lbl_col, del_exist=True):
        '''
        处理cols_to_label_info中已经存在的待标注列，
        del_exist为True时删除重复的
        '''
        if col in cols_to_label_info.keys():
            if  len(cols_to_label_info[col]) > 0 and del_exist:
                for k in range(len(cols_to_label_info[col])):
                    if cols_to_label_info[col][k][0] == lbl_col:
                        del cols_to_label_info[col][k]
        else:
            cols_to_label_info[col] = []
    
    for col, lbl_infos in conlabel_info.items():
        lbl_infos_new = []
        for lbl_info in lbl_infos:
            lbl_col = lbl_info[0]
            deal_exist_lbl_col(col, lbl_col, del_exist=del_repeat_lbl)
            Nval = len(lbl_info[1])
            tmp = 0
            for k in range(0, Nval):
                val = lbl_info[1][k]
                start_ends = get_con_start_end(df_[lbl_col], lambda x: x == val)
                for _ in range(0, len(start_ends)):
                    new_col = '_'+lbl_col+'_tmp_'+str(tmp)+'_'
                    df_[new_col] = np.nan
                    idx0, idx1 = start_ends[_][0], start_ends[_][1]+1
                    df_.loc[df_.index[idx0: idx1], new_col] = val
                    if _ == 0:
                        lbl_infos_new.append([new_col, (val,),
                                      (lbl_info[2][k],), (lbl_info[3][k],)])
                    else:
                        lbl_infos_new.append([new_col, (val,),
                                      (lbl_info[2][k],), (False,)])
                    tmp += 1
        # cols_to_label_info[col] += lbl_infos_new
        cols_to_label_info[col] = lbl_infos_new + cols_to_label_info[col]
        
    kwargs_new['cols_to_label_info'] = cols_to_label_info
    
    plot_Series(df_, **kwargs_new)
    
    
def plot_MaxMins(data, col, col_label, label_legend=['Max', 'Min'],
                 figsize=(11, 6), grid=True, title=None, nXticks=8,
                 markersize=10, fontsize=15, fig_save_path=None, **kwargs):
    '''
    绘制序列数据（data[col指定列]）并标注极大极小值点，idx1和idx2指定绘图用数据起止位置
    data必须包含列: [col指定列, col_label指定列]
    col_label指定列中值1表示极大值点，-1表示极小值点，0表示普通点
    label_legend指定col_label为1和-1时的图标标注
    nXticks设置x轴刻度显示数量
    **kwargs可输入plot_Series支持的其它参数
    '''
    plot_Series(data, {col: ('-k.', None)},
                cols_to_label_info={col: [[col_label, (1, -1), ('bv', 'r^'),
                                          label_legend]]},
                grids=grid, figsize=figsize, title=title, nXticks=nXticks,
                markersize=markersize, fontsize=fontsize, 
                fig_save_path=fig_save_path, **kwargs)
            
            
def plot_MaxMins_bk(data, col, col_label, label_legend=['Max', 'Min'],
                    figsize=(11, 6), grid=True, title=None, nXticks=8,
                    markersize=10, fontsize=15, fig_save_path=None):
    '''
    绘制序列数据（data[col指定列]）并标注极大极小值点，idx1和idx2指定绘图用数据起止位置
    data必须包含列: [col指定列, col_label指定列]
    col_label指定列中值1表示极大值点，-1表示极小值点，0表示普通点
    label_legend指定col_label为1和-1时的图标标注
    nXticks设置x轴刻度显示数量
    '''
    
    df = data.copy()
    if df.index.name is None:
        df.index.name = 'idx'        
    idx_name = df.index.name
    if idx_name in df.columns:
        df.drop(idx_name, axis=1, inplace=True)
    df.reset_index(inplace=True)
    
    series = df[col]
    series_max = df[df[col_label] == 1][col]
    series_min = df[df[col_label] == -1][col]
    
    plt.figure(figsize=figsize)
    plt.plot(series, '-k.', label=col)
    plt.plot(series_max, 'bv', markersize=markersize, label=label_legend[0])   
    plt.plot(series_min, 'r^', markersize=markersize, label=label_legend[1])
    plt.legend(loc=0, fontsize=fontsize)    
    
    n = df.shape[0]
    xpos = [int(x*n/nXticks) for x in range(0, nXticks)] + [n-1]
    plt.xticks(xpos, [df.loc[x, idx_name] for x in xpos])
    
    plt.grid(grid)
    
    if title:
        plt.title(title, fontsize=fontsize)
        
    if fig_save_path:
        plt.savefig(fig_save_path)
        
    plt.show()
    
    
if __name__ == '__main__':
    col1 = np.random.normal(10, 5, (100, 1))
    col2 = np.random.rand(100, 1)
    col3 = np.random.uniform(0, 20, (100, 1))
    col4 = col1 ** 2
    
    df = pd.DataFrame(np.concatenate((col1, col2, col3, col4), axis=1))
    df.columns = ['col1', 'col2', 'col3', 'col4']
    df['label1'] = df['col1'].apply(lambda x: 1 if x > 15 else \
                                                        (-1 if x < 5 else 0))
    df['label2'] = df['col3'].apply(lambda x: 1 if x > 15 else \
                                                        (-1 if x < 5 else 0))
    df.index = list(map(lambda x: 'idx'+str(x), df.index))
    
    
    plot_MaxMins(df, 'col1', 'label1', label_legend=['high', 'low'],
                 figsize=(11, 7), grid=False, title='col1', nXticks=20,
                 markersize=10, fontsize=20, fig_save_path=None)
        
        
    plot_Series(df, {'col1': ('.-r', None)},
                cols_styl_up_right={'col2': ('.-y', 0),
                                    'col3': ('-3', '')},
                # cols_styl_low_left={'col1': ('.-r', 't1')},
                cols_styl_low_right={'col4': ('.-k', 't4')},
                cols_to_label_info={'col2':
                                [['label1', (1, -1), ('gv', 'r^'), None]],
                                    'col4':
                                [['label2', (-1, 1), ('b*', 'mo'), None]]},
                xparls_info={'col1': [(10, 'k', '--', 3), (15, 'b', '-', 1)],
                              'col4': [(200, None, None, None)]},
                yparls_info_up=[('idx20', None, None, None),
                                ('idx90', 'g', '-', 4)],
                yparls_info_low=[('idx50', None, None, None),
                                  ('idx60', 'b', '--', 2)],
                ylabels=[False, '2', None, False],
                grids=False, figsize=(10, 8), title='test', nXticks=10,
                fontsize=20, markersize=10, 
                fig_save_path='./plot_test/plot_common.png', logger=None)
    plot_Series(df, {'col1': ('.-r', None)},
                cols_to_label_info={'col1': [['label1', (1, -1), ('gv', 'r^'),
                           None], ['label2', (-1, 1), ('*', 'o'), None]]},
                xparls_info={'col1': [(10, 'k', '--', 3), (15, 'b', '-', 1)],
                             'col4': [(200, None, None, None)]},
                yparls_info_up=[('idx20', None, None, None),
                                ('idx90', 'g', '-', 4)],
                yparls_info_low=[('idx50', None, None, None),
                                 ('idx60', 'b', '--', 2)],
                ylabels=['a', '2', None, False],
                grids=False, figsize=(10, 8), title='test', nXticks=10,
                fontsize=20, markersize=10, 
                fig_save_path='./plot_test/plot_common.png', logger=None)
    
    