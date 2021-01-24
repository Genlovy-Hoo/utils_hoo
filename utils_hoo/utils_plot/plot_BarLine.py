# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['font.serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_BarsLines(df, bar_col_infos_left, line_col_infos_left=None,
                   line_col_infos_right=None,
                   label_bar_data=False, label_line_data=False,
                   figsize=(11, 6), title=None, fontsize=15, 
                   fig_save_path=None, xtick_rotation=360):
    '''
    绘制条形图叠加折线图
    
    bar_col_infos_格式:
        {col1: (clor1, lbl1), col2: (clor2, lbl2), ...}
    line_col_infos_格式:
        {col1: (styl1, lbl1), col2: (styl2, lbl2), ...}
    '''
        
    def get_cols_clors_lbls(bar_cols_info, lbl_ext=None):
        cols, clors, lbls = [], [], []
        for col, clor_lbl in bar_cols_info.items():
            cols.append(col)
            clors.append(clor_lbl[0])
            lbls.append(clor_lbl[1]+lbl_ext if lbl_ext else clor_lbl[1])
        return cols, clors, lbls
    
    def add_bar_data(bars, ax):
        '''bar上标注数据'''
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, height,
                    ha='center', va='bottom', fontsize=fontsize)
            bar.set_edgecolor('white')
            
    def add_line_data(series, ax):
        '''在series（pd.Series）绘制的折线图上标注数据'''
        for idx, val in series.to_dict().items():
            ax.text(idx, val, val, ha='center', va='bottom', fontsize=fontsize)
    
    _, ax1 = plt.subplots(figsize=figsize)
    if line_col_infos_right is not None:
        ax2 = ax1.twinx()
    
    # 左轴bar
    bar_cols_left, bar_clors_left, bar_lbls_left = \
                                       get_cols_clors_lbls(bar_col_infos_left)
    bars = df[bar_cols_left].plot(ax=ax1, kind='bar', color=bar_clors_left)
    # 数据标注
    if label_bar_data:
        add_bar_data(bars.patches, ax1)
    
    # lns存放legends信息
    lns = [mpatches.Patch(color=bar_clors_left[_], label=bar_lbls_left[_]) \
                                      for _ in range(len(bar_col_infos_left))]
        
    # 左轴折线
    if line_col_infos_left is not None:
        for col, styl_lbl in line_col_infos_left.items():
            ln = ax1.plot(df[col], styl_lbl[0], label=styl_lbl[1])
            lns.append(ln[0])
            # 数据标注
            if label_line_data:
                add_line_data(df[col], ax1)
    
    # 右轴折线
    if line_col_infos_right is not None:
        for col, styl_lbl in line_col_infos_right.items():
            ln = ax2.plot(df[col], styl_lbl[0], label=styl_lbl[1]+'(r)')
            lns.append(ln[0])
            # 数据标注
            if label_line_data:
                add_line_data(df[col], ax2)

    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0, fontsize=fontsize)
    
    ax1.tick_params(labelsize=fontsize, rotation=xtick_rotation)
    
    plt.tight_layout()
    
    if title:
        plt.title(title, fontsize=fontsize)
        
    if fig_save_path:
        plt.savefig(fig_save_path)
    
    plt.show()


if __name__ == '__main__':
    df = pd.DataFrame({'test': [1, 2, 3, 4, 2, 3, 1, 6, 8, 2, 9]})
    df.index = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']
    df['rate'] = df['test'] / df['test'].sum()
    df['rate2'] = 1 - df['rate']
    df['test2'] = df['test'].apply(lambda x: x + np.random.randint(5))
    df['test3'] = df['test'].apply(lambda x: x + np.random.randint(10))
    
    bar_col_infos_left = {'test': ('g', 'test'), 'test2': ('b', 'test2')}
    
    line_col_infos_left = {'test3': ('c', 'test3')}
    line_col_infos_right = {'rate': ('.-r', 'rate'), 'rate2': ('.-m', 'rate2')}
    
    label_bar_data, label_line_data = True, True
    figsize, fontsize = (11, 6), 15
    
    plot_BarsLines(df, bar_col_infos_left=bar_col_infos_left,
                    line_col_infos_left=line_col_infos_left,
                   line_col_infos_right=line_col_infos_right,
                   label_bar_data=label_bar_data,
                   label_line_data=label_line_data,
                   figsize=figsize, fontsize=fontsize)
    
    
    