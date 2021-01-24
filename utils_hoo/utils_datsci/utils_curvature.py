# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import numpy.linalg as LA

import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['font.serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt


def Curvature_3Point(x, y):
    '''
    根据三个离散点计算曲率
    
    Parameters
    ----------
    x: 三个点x轴坐标列表
    y: 三个点y轴坐标列表
    
    Returns
    -------
    curvature: 曲率大小
    dircts: 曲率方向（标准化）
    
    参考：
        https://github.com/Pjer-zhang/Curvature_3Point
        https://zhuanlan.zhihu.com/p/72083902
    '''
    
    t_a = LA.norm([x[1]-x[0], y[1]-y[0]])
    t_b = LA.norm([x[2]-x[1], y[2]-y[1]])
    
    M = np.array([[1, -t_a, t_a**2],
                  [1, 0, 0],
                  [1, t_b, t_b**2]])

    a = np.matmul(LA.inv(M), x)
    b = np.matmul(LA.inv(M), y)

    curvature = 2 * (a[2]*b[1] - b[2]*a[1]) / (a[1]**2.0 + b[1]**2.0) ** 1.5
    dircts = [b[1], -a[1]] / np.sqrt(a[1]**2.0 + b[1]**2.0)
    
    return curvature, dircts


def rolling_Curvature_3Point(data, xcol, ycol, n_pre=3, n_post=3, gap=1):
    '''
    三点法滚动计算series（pd.Series）曲率
    data（pd.DataFrame）应包含[xcol, ycol]列
    返回df包含['curvature', 'dirct_x', 'dirct_y']三列
    '''
    
    df = data.reindex(columns=[xcol, ycol])
    df['curvature'] = np.nan
    df['dirct_x'] = np.nan
    df['dirct_y'] = np.nan
    
    for k in range(n_pre, df.shape[0]-n_post, gap):
        x = [df.loc[df.index[k-n_pre], xcol], df.loc[df.index[k], xcol],
             df.loc[df.index[k+n_post], xcol]]
        y = [df.loc[df.index[k-n_pre], ycol], df.loc[df.index[k], ycol],
             df.loc[df.index[k+n_post], ycol]]
        
        curvature, dircts = Curvature_3Point(x, y)
        
        df.loc[df.index[k], 'curvature'] = curvature
        df.loc[df.index[k], 'dirct_x'] = dircts[0]
        df.loc[df.index[k], 'dirct_y'] = dircts[1]
        
    return df[['curvature', 'dirct_x', 'dirct_y']]


def plot_curvature(data, cols, ax=None, plot_xy=True, std_dirt=False,
                   ax_equal=True, quvier_clor=None, xyline_styl=None,
                   figsize=(9, 6)):
    '''
    绘制曲率
    data中的cols应包含绘制曲率所需的全部列，包括：
        [x坐标列, y坐标列, 曲率值列, x方向列, y方向列]
    '''
    
    df = data.reindex(columns=cols)
    df.columns = ['x', 'y', 'curvature', 'dirct_x', 'dirct_y']
    
    if not std_dirt:
        df['dirct_x'] = df['curvature'] * df['dirct_x']
        df['dirct_y'] = df['curvature'] * df['dirct_y']
    
    if ax is None:
        _, ax = plt.subplots(figsize=figsize, dpi=120)
        
    if plot_xy:
        if xyline_styl is not None:
            ax.plot(df['x'], df['y'], xyline_styl)
        else:
            ax.plot(df['x'], df['y'])
    ax.quiver(df['x'], df['y'], df['dirct_x'], df['dirct_y'],
              color=quvier_clor)
    
    if ax_equal:
        ax.axis('equal')
    
    if ax is None:
        plt.show()
    else:
        return ax


if __name__ == '__main__':
    #%%
    # 圆
    df = pd.DataFrame({'theta': np.linspace(0, np.pi*2+0.2, 64)})
    df['x'] = 5 * np.cos(df['theta'])
    df['y'] = 5 * np.sin(df['theta'])
    
    n_pre = 1
    n_post = 1
    
    results = rolling_Curvature_3Point(df, 'x', 'y', n_pre, n_post)
    df = pd.merge(df, results, how='left', left_index=True, right_index=True)
    
    _, ax = plt.subplots(figsize=(6, 6), dpi=120)
    ax = plot_curvature(df, ['x', 'y', 'curvature', 'dirct_x', 'dirct_y'], ax, 
                        plot_xy=False, std_dirt=True, ax_equal=False)
    plt.show()
    
    #%%
    # 正弦曲线
    df = pd.DataFrame({'theta': np.linspace(0, np.pi*2 + 0.2, 64)})
    df['x'] = 5 * (df['theta'])
    df['y'] = 5 * np.sin(df['theta'])
    
    n_pre = 1
    n_post = 1
    
    results = rolling_Curvature_3Point(df, 'x', 'y', n_pre, n_post)
    df = pd.merge(df, results, how='left', left_index=True, right_index=True)

    ax = plot_curvature(df, ['x', 'y', 'curvature', 'dirct_x', 'dirct_y'],
                        figsize=(6, 4), quvier_clor='r', xyline_styl='.-k')
    
