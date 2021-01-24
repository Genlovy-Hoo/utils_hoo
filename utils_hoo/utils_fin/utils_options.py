# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import norm
from math import log, sqrt, exp


def BS_opt(Pnow, Pexe, D, r=3/100, sigma=22.5/100, D_year=365):
    '''
    BS期权定价公式
    
    Parameters
    ----------
    Pnow: 现价
    Pexe: 执行价
    D: 剩余天数
    r: 无风险利率，年化
    sigma: 波动率（目标标的收益波动率，年化）？
    D_year: 一年天数，默认为365，即自然天数，也可以只考虑交易天数
    
    Returns
    -------
    Popt_call: 认购做多期权合约价格
    Popt_put: 认沽做空期权合约价格
    
    参考：
    http://www.rmmsoft.com.cn/RSPages/onlinetools/OptionAnalysis/OptionAnalysisCN.aspx
    https://blog.csdn.net/qq_41239584/article/details/83383780
    https://zhuanlan.zhihu.com/p/38293827
    https://zhuanlan.zhihu.com/p/38294971
    https://zhuanlan.zhihu.com/p/96431951
    '''
    
    T = D / D_year
    
    d1 = (log(Pnow / Pexe) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    
    # 看涨
    Popt_call = Pnow * norm.cdf(d1) - Pexe * exp(-r * T) * norm.cdf(d2)
    # 看跌
    Popt_put = Pexe * exp(-r * T) * norm.cdf(-d2) - Pnow * norm.cdf(-d1)
    
    return Popt_call, Popt_put


def MCBS_opt(Pnow, Pexe, D, r=3/100, sigma=22.5/100, D_year=365,
             D_cut=60, N_mc=500000, random_seed=62):
    '''
    MC-BS，蒙特卡罗模拟BS公式计算期权价格
    
    Parameters
    ----------
    Pnow: 现价
    Pexe: 执行价
    D: 剩余天数
    r: 无风险利率，年化
    sigma: 波动率（目标标的收益波动率，年化）
    D_year: 一年天数，默认为365，即自然天数，也可以只考虑交易天数
    D_cut: 将一天划分为D_cut个小时段
    N_mc: 蒙特卡罗模拟次数
    random_seed: 随机数种子
    
    Returns
    -------
    Popt_call: 认购做多期权合约价格
    Popt_put: 认沽做空期权合约价格
    
    参考：    
    https://blog.csdn.net/qwop446/article/details/88914401
    https://blog.csdn.net/hzk427/article/details/104538974
    '''
    
    np.random.seed(random_seed)
    
    T = D / D_year
    dt = T / D_cut
    
    P = np.zeros((D_cut+1, N_mc)) # 模拟价格
    P[0] = Pnow
    for t in range(1, D_cut+1):
        db = np.random.standard_normal(N_mc) # 布朗运动随机游走
        P[t] = P[t-1] * np.exp((r - 0.5 * sigma ** 2) * dt + \
                                                        sigma * sqrt(dt) * db)
            
    # 看涨
    Popt_call = exp(-r * T) * np.sum(np.maximum(P[-1] - Pexe, 0)) / N_mc
    # 看跌
    Popt_put = exp(-r * T) * np.sum(np.maximum(Pexe - P[-1], 0)) / N_mc
    
    return Popt_call, Popt_put


def MCBSlog_opt(Pnow, Pexe, D, r=3/100, sigma=22.5/100, D_year=365,
                D_cut=60, N_mc=500000, random_seed=62):
    '''
    MC-BS，蒙特卡罗模拟BS公式计算期权价格，对数格式
    
    Parameters
    ----------
    Pnow: 现价
    Pexe: 执行价
    D: 剩余天数
    r: 无风险利率，年化
    sigma: 波动率（目标标的收益波动率，年化）
    D_year: 一年天数，默认为365，即自然天数，也可以只考虑交易天数
    D_cut: 将一天划分为D_cut个小时段
    N_mc: 蒙特卡罗模拟次数
    random_seed: 随机数种子
    
    Returns
    -------
    Popt_call: 认购做多期权合约价格
    Popt_put: 认沽做空期权合约价格
    
    参考：    
    https://blog.csdn.net/qwop446/article/details/88914401
    https://blog.csdn.net/hzk427/article/details/104538974
    '''
    
    np.random.seed(random_seed)
    
    T = D / D_year
    dt = T / D_cut
    
    dbs = np.random.standard_normal((D_cut+1, N_mc)) # 布朗运动随机游走
    dbs = np.cumsum((r - 0.5 * sigma**2) * dt + sigma * sqrt(dt) * dbs, axis=0)
    P = Pnow * np.exp(dbs) # 模拟价格
    
    # 看涨
    Popt_call = exp(-r * T) * np.sum(np.maximum(P[-1] - Pexe, 0)) / N_mc
    # 看跌
    Popt_put = exp(-r * T) * np.sum(np.maximum(Pexe - P[-1], 0)) / N_mc
    
    return Popt_call, Popt_put
    

if __name__ == '__main__':
    # BS公式&蒙特卡罗测试--------------------------------------------------------
    Pnow = 100
    Pexe = 105
    D = 120
    r = 2/100
    sigma = 30/100
    D_year = 365
    N_mc = 100000
    random_seed = None
    
    
    Popt_bs = BS_opt(Pnow, Pexe, D, r=r, sigma=sigma, D_year=D_year)
    Popt_mc = MCBS_opt(Pnow, Pexe, D, r=r, sigma=sigma, D_year=D_year,
                       N_mc=N_mc, random_seed=random_seed)
    Popt_mc_log = MCBSlog_opt(Pnow, Pexe, D, r=r, sigma=sigma, D_year=D_year,
                              N_mc=N_mc, random_seed=random_seed)
    print('\n')
    print(f'Popt_bs: {round(Popt_bs[0], 6)}')
    print(f'Popt_mc: {round(Popt_mc[0], 6)}')
    print(f'Popt_mclog: {round(Popt_mc_log[0], 6)}')
    print('\n')
    print(f'Popt_bs: {round(Popt_bs[1], 6)}')
    print(f'Popt_mc: {round(Popt_mc[1], 6)}')
    print(f'Popt_mclog: {round(Popt_mc_log[1], 6)}')
    