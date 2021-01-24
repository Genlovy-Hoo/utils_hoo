# -*- coding: utf-8 -*-

import time
import numpy as np
import pandas as pd
from math import sqrt
from utils_hoo import load_csv
from utils_hoo.utils_general import x_div_y, isnull, x_div_y, check_l_in_l0
from utils_hoo.utils_plot.plot_Common import plot_Series

#%%
def cal_beta():
    '''
    计算贝塔系数
    https://blog.csdn.net/thfyshz/article/details/83443783
    '''
    raise NotImplementedError
    # s1 = df1['pct_change']
    # s2 = df2['pct_change']
    # print((np.cov(s1, s2))[0][1]/np.var(s2))

#%%
def cal_gain_pct_log(Pcost, P, vP0=1):
    '''计算对数收益率Pcost为成本，P为现价，vP0为成本Pcost为0时的返回值'''
    if Pcost == 0:
        return vP0
    elif Pcost > 0:
        return np.log(P) - np.log(Pcost)
    else:
        raise ValueError('Pcost必须大于等于0！')


def cal_gain_pct(Pcost, P, vP0=1):
    '''
    计算盈亏比例
    Pcost为成本，P为现价，vP0为成本Pcost为0时的返回值
    
    注意：默认以权利方成本Pcost为正（eg.: 买入价为100，则Pcost=100）
         义务方成本Pcost为负进行计算（eg.: 卖出价为100，则Pcost=-100）
    '''
    if Pcost > 0:
        return P / Pcost - 1
    elif Pcost < 0:
        return 1 - P / Pcost
    else:
        return vP0
    
    
def cal_gain_pcts(Pseries, log=True):
    '''计算资产价值序列Pseries（pd.Series）每个时间的收益率'''
    if not log:
        return Pseries.pct_change()
    else:
        return Pseries.apply(np.log).diff()
    
    
def get_gains_prod(pct_series):
    '''累乘法计算累计收益率，pct_series为收益率（%）序列，pd.Series'''
    return (1 + pct_series / 100).cumprod()
    

def get_gains_sum(pct_series):
    '''累加法计算累计收益率，pct_series为收益率（%）序列，pd.Series'''
    return (pct_series / 100).cumsum()


def get_gains_act(df_settle):
    '''
    根据资金转入转出和资产总值记录计算实际总盈亏%（累计收益/累计投入）
    df_settle须包含列['转入', '转出', '资产总值']
    '''
    df = df_settle.reindex(columns=['转入', '转出', '资产总值'])
    df['净流入'] = df['转入'] - df['转出']
    df['累计投入'] = df['净流入'].cumsum()
    df['累计盈亏'] = df['资产总值'] - df['累计投入']
    df['实际总盈亏%'] = 100 * df[['累计投入', '累计盈亏']].apply(lambda x:
              x_div_y(x['累计盈亏'], x['累计投入'], v_xy0=0.0, v_y0=1.0), axis=1)
    # return df['实际总盈亏%']
    return df.reindex(columns=['累计投入', '累计盈亏', '实际总盈亏%'])


def get_gains_fundnet(df_settle):
    '''
    用基金净值法根据转入转出和资产总值记录计算净值
    df_settle须包含列['转入', '转出', '资产总值']
    '''
    df = df_settle.reindex(columns=['转入', '转出', '资产总值'])
    df['净流入'] = df['转入'] - df['转出']
    df['份额'] = np.nan
    df['净值'] = np.nan
    for k in range(0, df.shape[0]):
        if k == 0:
            df.loc[df.index[k], '新增份额'] = df.loc[df.index[k], '净流入']
            df.loc[df.index[k], '份额'] = df.loc[df.index[k], '新增份额']
        else:
            df.loc[df.index[k], '新增份额'] = df.loc[df.index[k], '净流入'] / \
                                                 df.loc[df.index[k-1], '净值']
            df.loc[df.index[k], '份额'] = df.loc[df.index[k-1], '份额'] + \
                                                df.loc[df.index[k], '新增份额']
        df.loc[df.index[k], '净值'] = df.loc[df.index[k], '资产总值'] / \
                                                   df.loc[df.index[k], '份额']
    return df.reindex(columns=['新增份额', '份额', '净值'])

    
def get_gains(df_settle, time_col='日期', gain_types=['act', 'fundnet']):
    '''
    不同类型的盈亏情况统计
    gain_types为累计收益计算方法，可选：
        ['act'实际总盈亏, 'prod'累乘法, 'sum'累加法, 'fundnet'基金份额净值法]
    注：累乘法和累加法df_settle须包含'盈亏%'列
       实际总盈亏和基金净值法df_settle须包含['转入', '转出', '资产总值']列   
    '''
    
    df_gain = df_settle.sort_values(time_col, ascending=True)
    df_gain.reset_index(drop=True, inplace=True)
    
    if 'act' in gain_types:
        cols = ['转入', '转出', '资产总值']
        if any([x not in df_settle.columns for x in cols]):
            raise ValueError('计算实际盈亏要求包含[`转入`, `转出`, `资产总值`]列！')
        df_act = get_gains_act(df_settle)
        df_gain = pd.merge(df_gain, df_act, how='left', left_index=True,
                           right_index=True)
    
    if 'prod' in gain_types:
        if not '盈亏%' in df_settle.columns:
            raise ValueError('累乘法要求包含`盈亏%`列！')
        df_gain['累乘净值'] = get_gains_prod(df_gain['盈亏%'])
    
    if 'sum' in gain_types:
        if not '盈亏%' in df_settle.columns:
            raise ValueError('累加法要求包含`盈亏%`列！')
        df_gain['累加净值'] = get_gains_sum(df_gain['盈亏%'])        
        
    if 'fundnet' in gain_types:
        cols = ['转入', '转出', '资产总值']
        if any([x not in df_settle.columns for x in cols]):
            raise ValueError('基金净值法要求包含[`转入`, `转出`, `资产总值`]列！')
        df_net = get_gains_fundnet(df_settle)
        df_gain = pd.merge(df_gain, df_net, how='left', left_index=True,
                           right_index=True)
    
    return df_gain.sort_values(time_col, ascending=False)


def plot_gain_act(df_gain, time_col='日期', N=None, **kwargs):
    '''
    绘制实际盈亏曲线图
    df_gain须包含列[time_col, '实际总盈亏%']
    **kwargs为plot_Series可接受参数
    '''
    
    N = df_gain.shape[0] if N is None or N < 1 or N > df_gain.shape[0] else N
    df = df_gain.reindex(columns=[time_col, '实际总盈亏%'])
    if N == df_gain.shape[0]:
        df.sort_values(time_col, ascending=True, inplace=True)
        tmp = pd.DataFrame(columns=['日期', '实际总盈亏%'])
        tmp.loc['tmp'] = ['start', 0]
        df = pd.concat((tmp, df), axis=0)
    else:
        df = df.sort_values(time_col, ascending=True).iloc[-N-1:, :]
        
    df.set_index(time_col, inplace=True)
    
    if not 'title' in kwargs:
        if N == df_gain.shape[0]:
            kwargs['title'] = '期权账户实际总盈亏(%)走势'
        else:
            kwargs['title'] = '期权账户近{}个交易日实际总盈亏(%)走势'.format(N)
    
    plot_Series(df, {'实际总盈亏%': '-ro'}, **kwargs)
    

def plot_gain_prod(df_gain, time_col='日期', N=None, show_gain=True, **kwargs):
    '''
    绘制盈亏净值曲线图
    df_gain须包含列[time_col, '盈亏%']
    **kwargs为plot_Series可接受参数
    '''
    
    N = df_gain.shape[0] if N is None or N < 1 or N > df_gain.shape[0] else N
    df = df_gain.reindex(columns=[time_col, '盈亏%'])
    if N >= df.shape[0]:
        df.sort_values(time_col, ascending=True, inplace=True)
        tmp = pd.DataFrame(columns=[time_col, '盈亏%'])
        tmp.loc['tmp'] = ['start', 0]
        df = pd.concat((tmp, df), axis=0)
    else:
        df = df.sort_values(time_col, ascending=True).iloc[-N-1:, :]    
        
    df.set_index(time_col, inplace=True)
    df.loc[df.index[0], '盈亏%'] = 0
    df['净值'] = (1 + df['盈亏%'] / 100).cumprod()
    gain_pct = round(100 * (df['净值'].iloc[-1]-1), 2)
    
    if not 'title' in kwargs:
        if N == df_gain.shape[0]:
            kwargs['title'] = f'期权账户净值曲线\n(收益率: {gain_pct}%)' \
                              if show_gain else '期权账户净值曲线'
        else:
             kwargs['title'] = \
                     f'期权账户近{N}个交易日净值曲线\n(收益率: {gain_pct}%)' \
                     if show_gain else f'期权账户近{N}个交易日净值曲线'
    
    plot_Series(df, {'净值': '-ro'}, **kwargs)

#%%
def cal_sig_gains(data, sig_col, VolF_add='base_1', VolF_sub='base_1',
                  VolF_stopLoss=0, IgnrSigNoStop=False, col_price='close',
                  col_price_buy='close', col_price_sel='close',
                  baseMny=200000, baseVol=None, fee=1.5/1000,
                  max_loss=1.0/100, max_gain=2.0/100, max_down=0.5/100):
    '''
    统计信号收益情况（A股）
    
    Parameters
    ----------
    data: 行情数据，其列须包含：
          sig_col列为信号列，其中1为做空（卖出）信号，-1为做多（买入）信号，0为不操作
          注：sig_col列的值只能包含-1和1和0
          col_price为结算价格列
          col_price_buy和col_price_sel分别为做多（买入）和做空（卖出）操作的价格列
    VolF_add: 自定义开仓/加仓操作时的交易量函数，其输入和输出格式应为：
        def VolF_add(baseVol, holdVol):
            # Parameters:
            #    baseVol：底仓量；
            #    holdVol：当前持仓量
            # Returns:
            # tradeVol：计划交易量    
            ......
            return tradeVol
        当VolF_add指定为'base_x'时，使用预定义函数get_AddTradeVol_baseX，
        其交易计划为：
            无持仓时开底仓，有持仓时开底仓的x倍
        当VolF_add指定为'hold_x'时，使用预定义函数get_AddTradeVol_holdX，
        其交易计划为：
            无持仓时开底仓，有持仓时开持仓的x倍
    VolF_sub: 自定义平仓/减仓操作时的交易量函数，其输入和输出格式应为：
        def VolF_sub(baseVol, holdVol):
            # Parameters:
            #    baseVol：底仓量；
            #    holdVol：当前持仓量
            # Returns:
            # tradeVol：计划交易量    
            ......
            return tradeVol
        当VolF_sub指定为'base_x'时，使用预定义函数get_SubTradeVol_baseX，
        其交易计划为：
            减底仓的x倍（若超过了持仓量相当于平仓后反向开仓）
        当VolF_sub指定为'hold_x'时，使用预定义函数get_SubTradeVol_holdX，
        其交易计划为：
            减持仓的x倍（x大于1时相当于平仓后反向开仓）
        当VolF_sub指定为'hold_base_x'时，使用预定义函数get_SubTradeVol_holdbaseX，
        其交易计划为：
            平仓后反向以baseVol的x倍反向开仓
    VolF_stopLoss: 自定义止损后的反向交易量函数，其输入和输出格式应为：
        def VolF_stopLoss(baseVol, holdVol):
            # Parameters:
            #    baseVol：底仓量；
            #    holdVol：当前持仓量
            # Returns:
            # tradeVol：计划交易量    
            ......
            return tradeVol
        当VolF_stopLoss指定为0或None或np.nan时，止损不反向开仓
        当VolF_stopLoss指定为'base_x'时，使用预定义函数get_stopLossTradeVol_baseX，
        其交易计划为：
            反向开底仓的x倍
        当VolF_stopLoss指定为'hold_x'时，使用预定义函数get_stopLossTradeVol_holdX，
        其交易计划为：
            反向开持仓的x倍
    IgnrSigNoStop: 当有持仓且没有触及止盈止损条件时是否忽略信号，
                   为True时忽略，为False时不忽略
    baseMny: 开底仓交易限额
    baseVol: 开底仓交易限量
        注：同时设置baseMny和baseVol时以baseMny为准
    fee: 单向交易综合成本比例（双向收费）
    max_loss: 止损比例
    max_gain: 止盈比例
    max_down: 平仓最大回撤比例
    
    Returns
    -------
    gain_stats: 包含['收益/最大占用比', '总收益', '总回收', '总投入', '最大占用']列
    df: 包含中间过程数据
    '''
    
    def get_baseVol(Price):
        '''计算底仓交易量'''
        if not isnull(baseMny):
            return 100 * (baseMny // (100 * Price))
        elif isnull(baseVol):
            raise ValueError('baseMny和baseVol必须设置一个！')
        else:
            return baseVol
        
    def get_AddTradeVol_baseX(baseVol, holdVol, x):
        '''无持仓则开底仓，有持仓则开底仓的x倍'''
        if abs(holdVol) == 0:
            return baseVol
        return baseVol * x
    
    def get_AddTradeVol_holdX(baseVol, holdVol, x):
        '''无持仓则开底仓，有持仓则开持仓的x倍'''
        if abs(holdVol) == 0:
            return baseVol
        return abs(holdVol * x)
    
    def get_SubTradeVol_baseX(baseVol, holdVol, x):
        '''减底仓的x倍（超过持仓即为反向开仓）'''
        return baseVol * x
    
    def get_SubTradeVol_holdX(baseVol, holdVol, x):
        '''减持仓的x倍（x大于1即为反向开仓）'''
        return abs(holdVol * x)
    
    def get_SubTradeVol_holdbaseX(baseVol, holdVol, x):
        '''平仓后反向开底仓的x倍'''
        return abs(holdVol) + baseVol * x
    
    def get_stopLossTradeVol_baseX(baseVol, holdVol, x):
        '''止损后反向开底仓的x倍'''
        return baseVol * x
    
    def get_stopLossTradeVol_holdX(baseVol, holdVol, x):
        '''止损后反向开持仓的x倍'''
        return abs(holdVol * x)
    
    def get_AddTradeVol(Price, VolF_add, hold_vol):
        '''开/加仓量计算'''
        baseVol = get_baseVol(Price)
        if isinstance(VolF_add, str) and 'base' in VolF_add:
            x = int(VolF_add.split('_')[-1])
            tradeVol = get_AddTradeVol_baseX(baseVol, hold_vol, x)
        elif isinstance(VolF_add, str) and 'hold' in VolF_add:
            x = int(VolF_add.split('_')[-1])
            tradeVol = get_AddTradeVol_holdX(baseVol, hold_vol, x)
        else:
            tradeVol = VolF_add(baseVol, hold_vol)
        return tradeVol
    
    def get_SubTradeVol(Price, VolF_sub, hold_vol):
        '''平/减仓量计算'''
        baseVol = get_baseVol(Price)
        if isinstance(VolF_sub, str) and\
                                    'base' in VolF_sub and 'hold' in VolF_sub:
            x = int(VolF_sub.split('_')[-1])
            tradeVol = get_SubTradeVol_holdbaseX(baseVol, hold_vol, x)
        elif isinstance(VolF_sub, str) and 'base' in VolF_sub:
            x = int(VolF_sub.split('_')[-1])
            tradeVol = get_SubTradeVol_baseX(baseVol, hold_vol, x)
        elif isinstance(VolF_sub, str) and 'hold' in VolF_sub:
            x = int(VolF_sub.split('_')[-1])
            tradeVol = get_SubTradeVol_holdX(baseVol, hold_vol, x)
        else:
            tradeVol = VolF_sub(baseVol, hold_vol)
        return tradeVol
    
    def get_stopLossTradeVol(Price, VolF_stopLoss, hold_vol):
        '''止损后反向开仓量计算'''
        baseVol = get_baseVol(Price)
        if VolF_stopLoss == 0 or isnull(VolF_stopLoss):
            tradeVol = 0
        elif isinstance(VolF_stopLoss, str) and 'base' in VolF_stopLoss:
            x = int(VolF_stopLoss.split('_')[-1])
            tradeVol = get_stopLossTradeVol_baseX(baseVol, hold_vol, x)
        elif isinstance(VolF_stopLoss, str) and 'hold' in VolF_stopLoss:
            x = int(VolF_stopLoss.split('_')[-1])
            tradeVol = get_stopLossTradeVol_holdX(baseVol, hold_vol, x)
        else:
            tradeVol = VolF_stopLoss(baseVol, hold_vol)
        return tradeVol
    
    def get_tradeVol(sig, act_stop, holdVol_pre, buyPrice, selPrice):
        '''
        根据操作信号、止盈止损信号、前持仓量和交易价格计算交易方向和计划交易量
        '''
        if sig == 0:
            if holdVol_pre == 0 or act_stop == 0:
                return 0, 0
            else:
                if holdVol_pre > 0: # 做多止损或止盈
                    if act_stop == 0.5: # 做多止损
                        stopLossTradeVol = get_stopLossTradeVol(selPrice,
                                                  VolF_stopLoss, holdVol_pre)
                        return 1, holdVol_pre + stopLossTradeVol
                    else: # 做多止盈
                        return 1, holdVol_pre
                elif holdVol_pre < 0: # 做空止损或止盈
                    if act_stop == -0.5: # 做空止损
                        stopLossTradeVol = get_stopLossTradeVol(buyPrice,
                                                  VolF_stopLoss, holdVol_pre)
                        return -1, abs(holdVol_pre) + stopLossTradeVol
                    else: # 做空止盈
                        return -1, abs(holdVol_pre) 
        elif holdVol_pre == 0:
            tradePrice = buyPrice if sig == -1 else selPrice
            tradeVol = get_AddTradeVol(tradePrice, VolF_add, holdVol_pre)
            return sig, tradeVol
        elif holdVol_pre > 0: # 持有做多仓位
            if sig == 1:
                if act_stop == 0:
                    if not IgnrSigNoStop: # 正常减/平做多仓位
                        selVol = get_SubTradeVol(selPrice, VolF_sub, holdVol_pre)
                        return sig, selVol
                    else: # 不触及止盈止损时忽略信号（不操作）
                        return 0, 0
                else: # 需要先止损或止盈后再开做空仓位
                    selVol = get_AddTradeVol(selPrice, VolF_add, 0)
                    if act_stop == 0.5: # 先止损后再开做空仓位
                        stopRevSelVol = get_stopLossTradeVol(selPrice,
                                                 VolF_stopLoss, holdVol_pre)
                        return sig, max(selVol, stopRevSelVol) + holdVol_pre
                    else: # 先止盈后再开做空仓位                        
                        return sig, selVol+holdVol_pre
            elif sig == -1:
                if act_stop == 0:
                    if not IgnrSigNoStop: # 正常加做多仓位
                        buyVol = get_AddTradeVol(buyPrice, VolF_add, holdVol_pre)
                        return sig, buyVol
                    else: # 不触及止盈止损时忽略信号（不操作）
                        return 0, 0
                else: # 需要先止损或止盈后再开做多仓位
                    buyVol = get_AddTradeVol(buyPrice, VolF_add, 0)
                    if act_stop == 0.5: # 先止损后再开做多仓位
                        stopRevSelVol = get_stopLossTradeVol(selPrice, 
                                                 VolF_stopLoss, holdVol_pre)
                        selVolAll = stopRevSelVol + holdVol_pre
                        if buyVol == selVolAll:
                            return 0, 0
                        elif buyVol > selVolAll:
                            return -1, buyVol-selVolAll
                        else:
                            return 1, selVolAll-buyVol
                    else: # 先止盈后再开做多仓位
                        if buyVol == holdVol_pre:
                            return 0, 0
                        elif buyVol > holdVol_pre:
                            return -1, buyVol-holdVol_pre
                        else:
                            return 1, holdVol_pre-buyVol
        elif holdVol_pre < 0: # 持有做空仓位
            if sig == 1:
                if act_stop == 0:
                    if not IgnrSigNoStop: # 正常加做空仓位
                        selVol = get_AddTradeVol(selPrice, VolF_add, holdVol_pre)
                        return sig, selVol
                    else: # 不触及止盈止损时忽略信号（不操作）
                        return 0, 0
                else: # 需要先止盈或止损后再开做空仓位
                    selVol = get_AddTradeVol(selPrice, VolF_add, 0)
                    if act_stop == -0.5: # 先止损再开做空仓位
                        stopRevBuyVol = get_stopLossTradeVol(buyPrice, 
                                                VolF_stopLoss, holdVol_pre)
                        buyVolAll = stopRevBuyVol + abs(holdVol_pre)
                        if selVol == buyVolAll:
                            return 0, 0
                        elif selVol > buyVolAll:
                            return 1, selVol-buyVolAll
                        else:
                            return -1, buyVolAll-selVol
                    else: # 先止盈再开做空仓位
                        if selVol == abs(holdVol_pre):
                            return 0, 0
                        elif selVol > abs(holdVol_pre):
                            return 1, selVol-abs(holdVol_pre)
                        else:
                            return -1, abs(holdVol_pre)-selVol
            elif sig == -1:
                if act_stop == 0:
                    if not IgnrSigNoStop: # 正常减/平做空仓位
                        buyVol = get_SubTradeVol(buyPrice, VolF_sub, holdVol_pre)
                        return sig, buyVol
                    else: # 不触及止盈止损时忽略信号（不操作）
                        return 0, 0
                else: # 需要先止盈或止损后再开做多仓位
                    buyVol = get_AddTradeVol(buyPrice, VolF_add, 0)
                    if act_stop == -0.5: # 先止损再开做多仓位
                        stopRevBuyVol = get_stopLossTradeVol(buyPrice,
                                                VolF_stopLoss, holdVol_pre)
                        return sig, max(buyVol, stopRevBuyVol) + \
                                    abs(holdVol_pre)
                    else: # 先止盈再开做多仓位
                        return sig, buyVol+abs(holdVol_pre)
    
    def buy_act(df, k, buy_price, buy_vol, hold_vol_pre, hold_cost_pre):
        '''买入操作记录'''
        df.loc[df.index[k], 'buyVol'] = buy_vol
        df.loc[df.index[k], 'holdVol'] = buy_vol + hold_vol_pre
        cashPut = buy_vol * buy_price * (1+fee)
        df.loc[df.index[k], 'cashPut'] = cashPut
        if hold_vol_pre >= 0: # 做多加仓或开仓
            df.loc[df.index[k], 'holdCost'] = hold_cost_pre + cashPut
        else:
            if buy_vol < abs(hold_vol_pre): # 减做空仓位
                df.loc[df.index[k], 'holdCost'] = hold_cost_pre + cashPut
            elif buy_vol > abs(hold_vol_pre): # 平做空仓位后反向开做多仓位
                df.loc[df.index[k], 'holdCost'] = \
                               (buy_vol + hold_vol_pre) * buy_price * (1+fee)
        return df
        
    def sel_act(df, k, sel_price, sel_vol, hold_vol_pre, hold_cost_pre):
        '''卖出操作记录'''
        df.loc[df.index[k], 'selVol'] = sel_vol
        df.loc[df.index[k], 'holdVol'] = hold_vol_pre - sel_vol
        cashGet = sel_vol * sel_price * (1-fee)
        df.loc[df.index[k], 'cashGet'] = cashGet
        if hold_vol_pre <= 0: # 做空加仓或开仓
            df.loc[df.index[k], 'holdCost'] = hold_cost_pre - cashGet
        else:
            if sel_vol < hold_vol_pre: # 减做多仓位
                df.loc[df.index[k], 'holdCost'] = hold_cost_pre - cashGet
            elif sel_vol > hold_vol_pre: # 平做多仓位后反向开做空仓位
                df.loc[df.index[k], 'holdCost'] = \
                                (hold_vol_pre-sel_vol) * sel_price * (1-fee)
        return df
        
    act_types = list(data[sig_col].unique())
    if not check_l_in_l0(act_types, [0, 1, -1]):
        raise ValueError(f'data.{sig_col}列的值只能是0或1或-1！')
        
    cols = list(set([sig_col, col_price, col_price_buy, col_price_sel]))
    df = data.reindex(columns=cols)
    
    df['buyVol'] = 0 # 做多（买入）量
    df['selVol'] = 0 # 做空（卖出）量
    df['holdVol'] = 0 # 持仓量（交易完成后）
    df['holdVal'] = 0 # 持仓价值（交易完成后）
    df['cashPut'] = 0 # 现金流出
    df['cashGet'] = 0 # 现金流入
    
    df['holdCost'] = 0 # 现有持仓总成本（交易完成后）
    df['holdPreGainPct'] = 0 # 持仓盈亏（交易完成前）
    df['holdPreGainPctMax'] = 0 # 持仓达到过的最高收益（交易完成前）
    df['holdPreMaxDown'] = 0 # 持仓最大回撤（交易完成前）
    df['act_stop'] = 0 # 止盈止损标注（0.5多止损，1.5多止盈，-0.5空止损，-1.5空止盈）
    df['act'] = df[sig_col] # 实际操作（1做空，-1做多）（用于信号被过滤时进行更正）
    df['holdGainPct'] = 0 # 现有持仓盈亏（交易完成后）
    
    last_act = 0 # 上一个操作类型
    for k in range(0, df.shape[0]):
        # 交易前持仓量
        if k == 0:
            holdVol_pre = 0
            holdCost_pre = 0
            act_stop = 0
            holdPreGainPct = 0
            holdPreGainPctMax = 0
            holdPreMaxDown = 0
        else:
            holdVol_pre = df.loc[df.index[k-1], 'holdVol']
            holdCost_pre = df.loc[df.index[k-1], 'holdCost']
            if holdVol_pre == 0:
                act_stop = 0
                holdPreGainPct = 0
                holdPreGainPctMax = 0
                holdPreMaxDown = 0
            else:            
                # 检查止盈止损是否触及  
                Price = df.loc[df.index[k], col_price]
                if holdVol_pre > 0:
                    holdVal_pre = holdVol_pre * Price * (1-fee)
                elif holdVol_pre < 0:
                    holdVal_pre = holdVol_pre * Price * (1+fee)
                holdPreGainPct = cal_gain_pct(holdCost_pre, holdVal_pre, vP0=0)
                # 若前一次有操作，则计算持仓盈利和回撤（交易前）须重新计算
                if last_act == 0:
                    holdPreGainPctMax = max(holdPreGainPct,
                                    df.loc[df.index[k-1], 'holdPreGainPctMax'])
                else:
                    holdPreGainPctMax = max(holdPreGainPct, 0)
                holdPreMaxDown = holdPreGainPctMax - holdPreGainPct
                # 没有止盈止损
                if isnull(max_loss) and isnull(max_gain) and isnull(max_down):
                    act_stop = 0
                # 固定比率止盈止损
                elif not isnull(max_loss) and not isnull(max_gain):
                    if holdPreGainPct <= -max_loss: # 止损
                        if holdCost_pre > 0:
                            act_stop = 0.5 # 做多止损
                        elif holdCost_pre < 0:
                            act_stop = -0.5 # 做空止损
                        else:
                            act_stop = 0
                    elif holdPreGainPct >= max_gain: # 止盈
                        if holdCost_pre > 0:
                            act_stop = 1.5 # 做多止盈
                        elif holdCost_pre < 0:
                            act_stop = -1.5 # 做空止盈
                        else:
                            act_stop = 0
                    else:
                        act_stop = 0
                # 最大回撤平仓
                elif not isnull(max_down):
                    if holdPreMaxDown < max_down:
                        act_stop = 0
                    else:
                        if holdCost_pre > 0:
                            # act_stop = 0.5 # 做多平仓
                            if holdPreGainPct < 0:
                                act_stop = 0.5 # 做多止损
                            elif holdPreGainPct > 0:
                                act_stop = 1.5 # 做多止盈
                            else:
                                act_stop = 0
                        elif holdCost_pre < 0:
                            # act_stop = -0.5 # 做空平仓
                            if holdPreGainPct < 0:
                                act_stop = -0.5 # 做空止损
                            elif holdPreGainPct > 0:
                                act_stop = -1.5 # 做空止盈
                            else:
                                act_stop = 0
                        else:
                            act_stop = 0
        
        df.loc[df.index[k], 'act_stop'] = act_stop
        df.loc[df.index[k], 'holdPreGainPct'] = holdPreGainPct
        df.loc[df.index[k], 'holdPreGainPctMax'] = holdPreGainPctMax
        df.loc[df.index[k], 'holdPreMaxDown'] = holdPreMaxDown
                
        buyPrice = df.loc[df.index[k], col_price_buy]
        selPrice = df.loc[df.index[k], col_price_sel]
        sig = df.loc[df.index[k], sig_col] # 操作信号
        
        # 确定交易计划
        if sig == 1:
            act, tradeVol = get_tradeVol(sig, act_stop, holdVol_pre,
                                         buyPrice, selPrice)
        elif sig == -1:
            act, tradeVol = get_tradeVol(sig, act_stop, holdVol_pre,
                                         buyPrice, selPrice)
        else:
            act, tradeVol = get_tradeVol(sig, act_stop, holdVol_pre,
                                         buyPrice, selPrice)
            
        # 更新被过滤信号实际操作
        if IgnrSigNoStop and sig != 0 and act == 0:
            df.loc[df.index[k], 'act'] = act
              
        # 交易执行
        if act == 0:
            df.loc[df.index[k], 'holdVol'] = holdVol_pre
            df.loc[df.index[k], 'holdCost'] = holdCost_pre
        elif act == -1:
            df = buy_act(df, k, buyPrice, tradeVol, holdVol_pre, holdCost_pre)
        elif act == 1:
            df = sel_act(df, k, selPrice, tradeVol, holdVol_pre, holdCost_pre)
            
        # 持仓信息更新
        holdVol = df.loc[df.index[k], 'holdVol']
        Price = df.loc[df.index[k], col_price]
        if holdVol > 0:
            df.loc[df.index[k], 'holdVal'] = holdVol * Price * (1-fee)
        elif holdVol < 0:
            df.loc[df.index[k], 'holdVal'] = holdVol * Price * (1+fee)
            
        df.loc[df.index[k], 'holdGainPct'] = cal_gain_pct(
           df.loc[df.index[k], 'holdCost'], df.loc[df.index[k], 'holdVal'], 0)
        
        last_act = act
                                
    df['cashGet_cum'] = df['cashGet'].cumsum() # 收入累计
    df['cashPut_cum'] = df['cashPut'].cumsum() # 支出累计
    df['gain_cum'] = df['cashGet_cum'] + df['holdVal'] - df['cashPut_cum'] # 累计盈利
    df['cashUsed'] = abs(df['cashPut_cum'] - df['cashGet_cum'])
    df['cashUsedMax'] = df['cashUsed'].cummax() # 最大资金占用
    df['pctGain_maxUsed'] = df[['gain_cum', 'cashUsedMax']].apply( lambda x:
      x_div_y(x['gain_cum'], x['cashUsedMax'], v_xy0=0), axis=1) # 收益/最大占用
    
    totalGet = df['cashGet_cum'].iloc[-1] + df['holdVal'].iloc[-1] # 总收入
    totalPut = df['cashPut_cum'].iloc[-1] # 总支出
    
    cashUsedMax = df['cashUsedMax'].iloc[-1]
    totalGain = df['gain_cum'].iloc[-1] # 总收益额    
    pctGain_maxUsed = df['pctGain_maxUsed'].iloc[-1]
    
    gain_stats = [pctGain_maxUsed, totalGain, totalGet, totalPut, cashUsedMax]
    gain_stats = pd.DataFrame(gain_stats).transpose()
    gain_stats.columns = ['收益/最大占用比', '总收益', '总回收', '总投入', '最大占用']
    
    return gain_stats, df
    
#%%
def cal_returns_period(Pseries, use_log=True, rtype='exp', N=252):
    '''
    计算周期化收益率
    
    Parameters
    ----------
    Pseries: 资产价值序列，pd.Series（不应有负值）
    use_log: 是否使用对数收益率，为False则使用普通收益率
    rtype: 周期化时采用指数方式'exp'或平均方式'mean'
    N: 一个完整周期包含的期数，eg.：
        若Pseries周期为日，求年化收益率时N一般为252（一年的交易日数）
        若Pseries周期为日，求月度收益率时N一般为21（一个月的交易日数）
        若Pseries周期为分钟，求年化收益率时N一般为252*240（一年的交易分钟数）
    
    Returns
    -------
    r: 周期化收益率，其周期由N确定
    '''
    
    Pseries = np.array(Pseries)
    nP = len(Pseries)
    
    if use_log:
        gain_pct = cal_gain_pct_log(Pseries[0], Pseries[-1])
    else:
        gain_pct = cal_gain_pct(Pseries[0], Pseries[-1])
    
    if rtype == 'exp':
        r = (1 + gain_pct) ** (N / nP) - 1
    elif rtype == 'mean':
        r = N * gain_pct / nP
        
    return r    

def cal_volatility(Pseries, use_log=True, N=252):
    '''
    价格序列Pseries周期化波动率计算
    
    Parameters
    ----------
    Pseries: 资产价值序列，pd.Series（不应有负值）
    use_log: 是否使用对数收益率，为False则使用普通收益率
    N: 一个完整周期包含的期数，eg.：
        若Pseries周期为日，求年化收益波动率时N一般为252（一年的交易日数）
        若Pseries周期为日，求月度收益波动率时N一般为21（一个月的交易日数）
        若Pseries周期为分钟，求年化收益波动率时N一般为252*240（一年的交易分钟数）
    
    Returns
    -------
    r: 收益波动率，其周期由N确定
    '''
    
    col = 'series'
    Pseries.name = col
    df = pd.DataFrame(Pseries)
    
    # 收益率
    df['gain_pct'] = cal_gain_pcts(Pseries, log=use_log)
        
    # 波动率
    r = df['gain_pct'].std() * sqrt(N)

    return r


def cal_sharpe(values, r=3/100, N=252, use_log=True):
    '''
    计算夏普比率
    values: 资产价值序列，pd.Series
    r: 无风险收益率
    N: 无风险收益率r的周期所包含的values的周期数，eg.：
        若values周期为日，r为年化无风险收益率时，N一般为252（一年的交易日数）
        若values周期为日，r月度无风险收益率时，N一般为21（一个月的交易日数）
        若values周期为分钟，r为年化无风险收益率时，N一般为252*240（一年的交易分钟数）
    use_log: 计算收益时是否采用对数形式
    
    https://www.joinquant.com/help/api/help?name=api#风险指标
    https://www.jianshu.com/p/363aa2dd3441
    '''
    df = pd.DataFrame({'values': values})
    df['gains'] = cal_gain_pcts(df['values'], log=use_log) # 收益率序列
    df['gains_ex'] = df['gains'] - r/N # 超额收益
    return sqrt(df.shape[0]) * df['gains_ex'].mean() / df['gains_ex'].std()
    
    
def get_MaxDown(values, return_idx=True):
    '''
    最大回撤计算
    
    Parameters
    ----------
    values: 资产价值序列，list或一维np.array或pd.Series
    
    Returns
    -------
    -maxDown: 最大回撤幅度（正值）
    (start_idx, end_idx): 最大回撤起止位置（int），若return_idx为False，则为None
    
    https://www.cnblogs.com/xunziji/p/6760019.html
    https://blog.csdn.net/changyan_123/article/details/80994170
    '''
    
    n = len(values)
    data = np.array(values)
    
    if not return_idx:
        maxDown, tmp_max = 0, -np.inf
        for k in range(1, n):
            tmp_max = max(tmp_max, data[k-1])
            maxDown = min(maxDown, data[k] / tmp_max - 1)
        return -maxDown, (None, None)
    else:
        Cmax, Cmax_idxs = np.zeros(n-1), [0 for _ in range(n-1)]
        tmp_max = -np.inf
        tmp_idx = 0
        for k in range(1, n):
            if data[k-1] > tmp_max:
                tmp_max =  data[k-1]
                tmp_idx = k-1
            Cmax[k-1] = tmp_max
            Cmax_idxs[k-1] = tmp_idx
        
        maxDown = 0.0
        start_idx, end_idx = 0, 0
        for k in range(1, n):
            tmp = data[k] / Cmax[k-1] - 1
            if tmp < maxDown:
                maxDown = tmp
                start_idx, end_idx = Cmax_idxs[k-1], k
                
        return -maxDown, (start_idx, end_idx)
    
    
    
def get_MaxUp(values, return_idx=True):
    '''
    最大盈利计算（与最大回撤相对应，即做空情况下的最大回撤）
    
    Parameters
    ----------
    values: 资产价值序列，list或一维np.array或pd.Series
    
    Returns
    -------
    maxUp: 最大盈利幅度（正值）
    (start_idx, end_idx): 最大回撤起止位置（int），若return_idx为False，则为None
    
    https://www.cnblogs.com/xunziji/p/6760019.html
    '''
    
    n = len(values)
    data = np.array(values)
    
    if not return_idx:
        maxUp, tmp_min = 0, np.inf
        for k in range(1, n):
            tmp_min = min(tmp_min, data[k-1])
            maxUp = max(maxUp, data[k] / tmp_min - 1)
        return maxUp, (None, None)
    else:
        Cmin, Cmin_idxs = np.zeros(n-1), [0 for _ in range(n-1)]
        tmp_min = np.inf
        tmp_idx = 0
        for k in range(1, n):
            if data[k-1] < tmp_min:
                tmp_min =  data[k-1]
                tmp_idx = k-1
            Cmin[k-1] = tmp_min
            Cmin_idxs[k-1] = tmp_idx
        
        maxUp = 0.0
        start_idx, end_idx = 0, 0
        for k in range(1, n):
            tmp = data[k] / Cmin[k-1] - 1
            if tmp > maxUp:
                maxUp = tmp
                start_idx, end_idx = Cmin_idxs[k-1], k
                
        return maxUp, (start_idx, end_idx)
    
    
def get_MaxDown_pd(series):
    '''
    最大回撤计算，使用pd
    
    Parameters
    ----------
    series: 资产价值序列（pd.Series）
    
    Returns
    -------
    maxDown: 最大回撤幅度（正值）
    (start_idx, end_idx): 最大回撤起止索引
    (start_iloc, end_iloc): 最大回撤起止位置（int）
    '''
    
    df = pd.DataFrame(series)
    df.columns = ['val']
    df['idx'] = range(0, df.shape[0])
    
    df['Cmax'] = df['val'].cummax()
    df['maxDown_now'] = df['val'] / df['Cmax'] - 1
    
    maxDown = -df['maxDown_now'].min()
    end_iloc = df['maxDown_now'].argmin()
    end_idx = df.index[end_iloc]
    start_idx = df[df['val'] == df.loc[df.index[end_iloc], 'Cmax']].index[0]
    start_iloc = df[df['val'] == df.loc[df.index[end_iloc], 'Cmax']]['idx'][0]
    
    return maxDown, (start_idx, end_idx), (start_iloc, end_iloc)

#%%
if __name__ == '__main__':
    strt_tm = time.time()
    
    #%%
    # 最大回撤测试
    values = [1.0, 1.01, 1.05, 1.1, 1.11, 1.07, 1.03, 1.03, 1.01, 1.02, 1.04,
              1.05, 1.07, 1.06, 1.05, 1.06, 1.07, 1.09, 1.12, 1.18, 1.15,
              1.15, 1.18, 1.16, 1.19, 1.17, 1.17, 1.18, 1.19, 1.23, 1.24,
              1.25, 1.24, 1.25, 1.24, 1.25, 1.24, 1.25, 1.24, 1.27, 1.23,
              1.22, 1.18, 1.2, 1.22, 1.25, 1.25, 1.27, 1.26, 1.31, 1.32, 1.31,
              1.33, 1.33, 1.36, 1.33, 1.35, 1.38, 1.4, 1.42, 1.45, 1.43, 1.46,
              1.48, 1.52, 1.53, 1.52, 1.55, 1.54, 1.53, 1.55, 1.54, 1.52,
              1.53, 1.53, 1.5, 1.45, 1.43, 1.42, 1.41, 1.43, 1.42, 1.45, 1.45,
              1.49, 1.49, 1.51, 1.54, 1.53, 1.56, 1.52, 1.53, 1.58, 1.58,
              1.58, 1.61, 1.63, 1.61, 1.59]
    data = pd.DataFrame(values, columns=['values'])
    # maxDown, (strt_idx, end_idx) = get_MaxDown(data['values'])
    maxDown, (strt_idx, end_idx) = get_MaxUp(data['values'])
    data['in_maxDown'] = 0
    data.loc[data.index[strt_idx: end_idx+1], 'in_maxDown'] = 1
    plot_Series(data, {'values': '.-b'},
                cols_to_label_info={'values':
                            [['in_maxDown', (1,), ('.-r',), ('最大回撤区间',)]]},
                grids=True, figsize=(11, 7))
    
    fpath = '../test/510050_daily_pre_fq.csv'
    data = load_csv(fpath)
    data.set_index('date', drop=False, inplace=True)
    data = data.iloc[:, :][['close']]
    
    plot_Series(data, {'close': '.-b'}, grids=True, figsize=(11, 7))
    
    maxDown, _ , (strt_idx, end_idx) = get_MaxDown_pd(data['close'])
    maxDown = str(round(maxDown, 4))
    data['in_maxDown'] = 0
    data.loc[data.index[strt_idx: end_idx+1], 'in_maxDown'] = 1
    plot_Series(data, {'close': '.-b'},
                cols_to_label_info={'close':
                            [['in_maxDown', (1,), ('.-r',), ('最大回撤区间',)]]},
                title='最大回撤：' + ' --> '.join(_) + ' (' + maxDown + ')',
                grids=True, figsize=(11, 7))
        
    # maxDown, (strt_idx, end_idx) = get_MaxDown(data['close'])
    maxDown, (strt_idx, end_idx) = get_MaxUp(data['close'])
    maxDown = str(round(maxDown, 4))
    _ = (data.index[strt_idx], data.index[end_idx])
    data['in_maxDown'] = 0
    data.loc[data.index[strt_idx: end_idx+1], 'in_maxDown'] = 1
    plot_Series(data, {'close': '.-b'},
                cols_to_label_info={'close':
                            [['in_maxDown', (1,), ('.-r',), ('最大回撤区间',)]]},
                title='最大回撤：' + ' --> '.join(_) + ' (' + maxDown + ')',
                grids=True, figsize=(11, 7))

    #%%
    # 收益率、波动率、夏普测试
    fpath = '../test/510050_daily_pre_fq.csv'
    data = load_csv(fpath)
    data.set_index('date', drop=False, inplace=True)
        
    # 年化收益
    print(f'50ETF年化收益率：{round(cal_returns_period(data.close), 6)}.')
    
    # 波动率
    n = 90
    volt = round(cal_volatility(data.close.iloc[-n:], False), 6)
    print(f'50ETF {n if n >0 else None} 日年化收益波动率：{volt}.')

    # 夏普
    n = 252
    values = data.close.iloc[-n:]
    sharpe = round(cal_sharpe(values, r=3/100, N=252, use_log=True), 6)
    print(f'50ETF {n if n >0 else None} 日年化夏普比率：{sharpe}.')
    
    #%%
    # 信号盈亏统计
    data['signal'] = 0
    for k in range(0, data.shape[0], 5):
        data.loc[data.index[k], 'signal'] = np.random.randint(-1, 2)
        
    sig_col = 'signal'
    VolF_add = 'hold_1'
    # VolF_add = lambda x, y: 2*x + 2*abs(y)
    # VolF_sub = 'hold_1'
    VolF_sub = 'hold_base_1'
    # VolF_sub = lambda x, y: 2*x + 2*abs(y)
    VolF_stopLoss = 'base_1'
    IgnrSigNoStop = True
    col_price = 'close'
    col_price_buy = 'close'
    col_price_sel = 'close'
    # baseMny = 15000
    baseMny = None
    baseVol = 10000
    fee = 1.5/1000
    
    # max_loss, max_gain, max_down = None, None, None # 不设止盈止损
    max_loss, max_gain, max_down = 1.0/100, 1.5/100, None # 盈亏比止盈止损
    # max_loss, max_gain, max_down = None, None, 2.0/100 # 最大回撤止盈止损
    
    data = data.iloc[-50:, :].copy()
    gain_stats, df = cal_sig_gains(data, sig_col,
                                   VolF_add=VolF_add,
                                   VolF_sub=VolF_sub,
                                   VolF_stopLoss=VolF_stopLoss,
                                   IgnrSigNoStop=IgnrSigNoStop,
                                   col_price=col_price,
                                   col_price_buy=col_price_buy,
                                   col_price_sel=col_price_sel,
                                   baseMny=baseMny, baseVol=baseVol,
                                   fee=fee, max_loss=max_loss,
                                   max_gain=max_gain, max_down=max_down)
    plot_Series(df, {'close': ('.-k', False)},
                cols_to_label_info={'close': 
                [['act', (-1, 1), ('r^', 'bv'), ('做多', '做空')],
                 ['act_stop', (-1.5, 1.5, -0.5, 0.5), ('r*', 'b*', 'mo', 'go'),
                  ('做空止盈', '做多止盈', '做空止损', '做多止损')]
                ]},
                markersize=15, figsize=(12, 7))
    
    print('\n')
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)
    print(gain_stats)
    
    #%%
    print(f'used time: {round(time.time()-strt_tm, 6)}s.')
    
    
    
    
    
    
    
    
    
    
    
    
    
    