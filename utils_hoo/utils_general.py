# -*- coding: utf-8 -*-

import logging
import numpy as np
import pandas as pd
from functools import reduce
from random import randint, random, uniform


def bootstrapping():
    '''
    bootstraping
    '''
    raise NotImplementedError


def Roulette_base(fitness):
    '''
    基本轮盘赌法
    fitness所有备选对象的fitness值列表，返回被选中对象的索引号
    注：fitness值应为正，且fitness值越大，被选中概率越大
    https://blog.csdn.net/armwangEric/article/details/50775206
    '''
    sumFits = sum(fitness)
    rndPoint = uniform(0, sumFits)
    accumulator = 0.0
    for idx, fitn in enumerate(fitness):
        accumulator += fitn
        if accumulator >= rndPoint:
            return idx
        
        
def Roulette_stochasticAccept(fitness):
    '''
    轮盘赌法，随机接受法
    fitness所有备选对象的fitness值列表，返回被选中对象的索引号
    注：fitness值应为正，且fitness值越大，被选中概率越大
    https://blog.csdn.net/armwangEric/article/details/50775206
    '''
    N = len(fitness)
    maxFitn = max(fitness)
    while True:
        idx = randint(0, N-1)
        if random() <= fitness[idx] / maxFitn:
            return idx
        
        
def Roulette_N(fitness, N=10000, randFunc=Roulette_stochasticAccept):
    '''
    轮盘赌法N次模拟，返回每个备选对象在N次模拟中被选中的次数
    N为模拟次数，randFunc指定轮盘赌法函数，如Roulette_base或Roulette_stochasticAccept
    fitness格式和返回格式对应如下（N取6000）：
        fitenss: [1, 2, 3]或(1, 2, 3)
        ——> [(0, 991), (1, 2022), (2, 2987)]
        fitness: [('a', 1), ('b', 2), ('c', 3)]或[['a', 1], ['b', 2], ['c', 3]]
        或(('a', 1), ('b', 2), ('c', 3))
        ——> [('a', 1016), ('b', 1921), ('c', 3063)]
        fitness: {'a': 1, 'b': 2, 'c': 3}
        ——> {'a': 988, 'b': 1971, 'c': 3041}
    注：fitness中的值应为正，且fitness值越大，被选中概率越大
    '''
    if isinstance(fitness, dict):
        Ks, Vs = [], []
        for k, v in fitness.items():
            Ks.append(k)
            Vs.append(v)
        randPicks = [randFunc(Vs) for _ in range(N)]
        idx_Picks = [(x, randPicks.count(x)) for x in range(len(Vs))]
        return {Ks[x[0]]: x[1] for x in idx_Picks}
    
    elif (isinstance(fitness[0], list) or isinstance(fitness[0], tuple)):
        Ks, Vs = [], []
        for k, v in fitness:
            Ks.append(k)
            Vs.append(v)
        randPicks = [randFunc(Vs) for _ in range(N)]
        idx_Picks = [(x, randPicks.count(x)) for x in range(len(Vs))]
        return [(Ks[x[0]], x[1]) for x in idx_Picks]
        
    elif (isinstance(fitness[0], int) or isinstance(fitness[0], float)):
        randPicks = [randFunc(fitness) for _ in range(N)]
        idx_Picks = [(x, randPicks.count(x)) for x in range(len(fitness))]
        return idx_Picks


def randSum(target_sum, n, lowests, highests, isInt=True, n_dot=6):
    '''
    在lowests和highests范围内随机产生n个数的列表adds，要求所选数之和为target_sum
    lowests和highests为实数或列表，指定每个备选数的上下限
    isInt若为True，则所选数全为整数，否则为实数，
    注：若输入lowests或highests不是int，则isInt为True无效
    n_dot，动态上下界值与上下限比较时控制小数位数（为了避免python精度问题导致的报错）
    '''
    
    if (not isinstance(lowests, int) and not isinstance(lowests, float)) \
                                                    and len(lowests) != n:
        raise ValueError('下限值列表（数组）lowests长度必须与n相等！')        
    if (not isinstance(highests, int) and not isinstance(highests, float)) \
                                                    and len(highests) != n:
        raise ValueError('上限值列表（数组）highests长度必须与n相等！')
        
    # lowests、highests组织成list
    if isinstance(lowests, int) or isinstance(lowests, float):
        lowests = [lowests] * n
    if isinstance(highests, int) or isinstance(highests, float):
        highests = [highests] * n
    
    if any([isinstance(x, float) for x in lowests]) or \
                            any([isinstance(x, float) for x in highests]):
        isInt = False
        
    LowHigh = list(zip(lowests, highests))
    
    def dyLowHigh(tgt_sum, low_high, n_dot=6):
        '''
        动态计算下界和上界，
        n_dot为小数保留位数（为了避免python精度问题导致的报错）
        '''
        restSumHigh = sum([x[1] for x in low_high[1:]])
        restSumLow = sum([x[0] for x in low_high[1:]])
        low = max(tgt_sum-restSumHigh, low_high[0][0])
        if round(low, n_dot) > low_high[0][1]:
            raise ValueError(
               '下界({})超过最大值上限({})！'.format(low, low_high[0][1]))
        high = min(tgt_sum-restSumLow, low_high[0][1])
        if round(high, n_dot) < low_high[0][0]:
            raise ValueError(
               '上界({})超过最小值下限({})！'.format(high, low_high[0][0]))
        return low, high
    
    S = 0
    adds = []
    low, high = dyLowHigh(target_sum, LowHigh, n_dot=n_dot)
    while len(adds) < n-1:
        # 每次随机选择一个数
        if isInt:
            randV = randint(low, high)
        else:
            randV = random() * (high-low) + low
        
        # 判断当前所选择的备选数是否符合条件，若符合则加入备选数，
        # 若不符合则删除所有备选数重头开始
        restSum = target_sum - (S + randV)
        restSumLow = sum([x[0] for x in LowHigh[len(adds)+1:]])
        restSumHigh = sum([x[1] for x in LowHigh[len(adds)+1:]])
        if restSumLow <= restSum <= restSumHigh:
            S += randV
            adds.append(randV)
            low, high = dyLowHigh(target_sum-S, LowHigh[len(adds):],
                                  n_dot=n_dot)
        else:
            S = 0
            adds = []            
            low, high = dyLowHigh(target_sum, LowHigh, n_dot=n_dot)
            
    adds.append(target_sum-sum(adds)) # 最后一个备选数
    
    return adds


def randWSum(weight_sum, n, lowests, highests, W=None, n_dot=6):
    '''
    在lowests和highests范围内随机产生n个数的列表adds，
    要求所选数之加权和为weight_sum，W为权重，若W为None，则等权
    lowests和highests为实数或列表，指定每个备选数的上下限
    注：lowests和highests与W应一一对应
    n_dot，动态上下界值与上下限比较时控制小数位数（为了避免python精度问题导致的报错）
    '''
    
    if W is not None and len(W) != n:
        raise ValueError('权重列表W的长度必须等于n！')        
    if (not isinstance(lowests, int) and not isinstance(lowests, float)) \
                                                    and len(lowests) != n:
        raise ValueError('下限值列表（数组）lowests长度必须与n相等！')        
    if (not isinstance(highests, int) and not isinstance(highests, float)) \
                                                    and len(highests) != n:
        raise ValueError('上限值列表（数组）highests长度必须与n相等！')
        
    # W和lowests、highests组织成list
    if W is None:
        W = [1/n] * n        
    if isinstance(lowests, int) or isinstance(lowests, float):
        lowests = [lowests] * n
    if isinstance(highests, int) or isinstance(highests, float):
        highests = [highests] * n
        
    WLowHigh = list(zip(W, lowests, highests))
    
    def dyLowHigh(wt_sum, w_low_high, n_dot=6):
        '''
        动态计算下界和上界，
        n_dot，动态上下界值与上下限比较时控制小数位数（为了避免python精度问题导致的报错）
        '''
        restSumHigh = sum([x[2]*x[0] for x in w_low_high[1:]])
        restSumLow = sum([x[1]*x[0] for x in w_low_high[1:]])
        low = max((wt_sum-restSumHigh) / w_low_high[0][0], w_low_high[0][1])
        if round(low, n_dot) > w_low_high[0][2]:
            raise ValueError(
               '下界({})超过最大值上限({})！'.format(low, w_low_high[0][2]))
        high = min((wt_sum-restSumLow) / w_low_high[0][0], w_low_high[0][2])
        if round(high, n_dot) < w_low_high[0][1]:
            raise ValueError(
               '上界({})超过最小值下限({})！'.format(high, w_low_high[0][1]))
        return low, high
    
    S = 0
    adds = []
    low, high = dyLowHigh(weight_sum, WLowHigh, n_dot=n_dot)
    while len(adds) < n-1:
        # 每次随机选择一个数
        randV = random() * (high-low) + low
        
        # 判断当前所选择的备选数是否符合条件，若符合则加入备选数，
        # 若不符合则删除所有备选数重头开始
        restSum = weight_sum - (S + randV * W[len(adds)])
        restSumLow = sum([x[1]*x[0] for x in WLowHigh[len(adds)+1:]])
        restSumHigh = sum([x[2]*x[0] for x in WLowHigh[len(adds)+1:]])
        if restSumLow <= restSum <= restSumHigh:
            S += randV * W[len(adds)]
            adds.append(randV)
            low, high = dyLowHigh(weight_sum-S, WLowHigh[len(adds):],
                                  n_dot=n_dot) 
        else:
            S = 0
            adds = []            
            low, high = dyLowHigh(weight_sum, WLowHigh, n_dot=n_dot)
            
    aw = zip(adds, W[:-1])
    adds.append((weight_sum-sum([a*w for a, w in aw])) / W[-1])
    
    return adds


def simple_logger():
    '''返回一个简单的logger（只在控制台打印日志信息）'''
    
    # 准备日志记录器logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    # 移除已有的handlers
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            h.close()
            logger.removeHandler(h)
    for h in logger.handlers:
        logger.removeHandler(h)
    
    # 日志格式
    formatter = logging.Formatter(
            '''%(asctime)s -%(filename)s[line: %(lineno)d] -%(levelname)s:
    --%(message)s''')
        
    # 控制台打印，StreamHandler
    console_logger = logging.StreamHandler()
    console_logger.setLevel(logging.DEBUG)
    console_logger.setFormatter(formatter)
    logger.addHandler(console_logger)
    
    return logger


def con_count(series, Fcond, via_pd=True):
    '''    
    计算series(pd.Series)中连续满足Fcond函数指定的条件的记录数
    Fcond为指定条件的函数，Fcond(x)返回结果只能为True或False
    若via_pd为False，则使用循环迭代，若via_pd为True，则使用pd
    返回pd.Series，即连续计数结果
    
    Examples
    --------
        df = pd.DataFrame([0, 0, 1, 1, 0, 0, 1, 1, 1], columns=['series'])
        Fcond = lambda x: True if x == 1 else False
        df['count1'] = con_count(df['series'], Fcond, True)
        df:
               series  count1
            0       0       0
            1       0       0
            2       1       1
            3       1       2
            4       0       0
            5       0       0
            6       1       1
            7       1       2
            8       1       3
        df['count0'] = con_count(df['series'], lambda x: x != 1, False)
        df:
               series  count1  count0
            0       0       0       1
            1       0       0       2
            2       1       1       0
            3       1       2       0
            4       0       0       1
            5       0       0       2
            6       1       1       0
            7       1       2       0
            8       1       3       0
    '''
    
    col = 'series'
    series.name = col
    df = pd.DataFrame(series)
    
    # 当series.index存在重复值时为避免报错，因此先重置index最后再还原
    ori_index = df.index
    df.index = range(0, df.shape[0])
    
    if via_pd:
        df['Fok'] = df[col].apply(lambda x: Fcond(x)).astype(int)
        df['count'] = df['Fok'].cumsum()
        df['tmp'] = df[df['Fok'] == 0]['count']
        df['tmp'] = df['tmp'].fillna(method='ffill')
        df['tmp'] = df['tmp'].fillna(0)
        df['count'] = (df['count'] - df['tmp']).astype(int)
        
        df.index = ori_index
        
        return df['count']
    
    else:
        df['count'] = 0
        k = 0
        while k < df.shape[0]:
            if Fcond(df.loc[df.index[k], col]):
                count = 1
                df.loc[df.index[k], 'count'] = count
                k1 = k + 1
                while k1 < df.shape[0] and Fcond(df.loc[df.index[k1], col]):
                    count += 1
                    df.loc[df.index[k1], 'count'] = count
                    k1 += 1
                k = k1
            else:
                k += 1
                
        df.index = ori_index
                
        return df['count']


def get_pre_val_Fcond(data, col_val, col_Fcond, Fcond):
    '''
    获取上一个满足指定条件的行中col_val列的值，条件为：该行中col_Fcond列的值x满足
    Fcond(x)为True（Fcond(x)返回结果只能为True或False）
    返回结果为pd.Series
    
    Examples
    --------
    data = pd.DataFrame({'x1': [0, 1, 1, 0, -1, -1, 2, -1, 1, 0, 1, 1, 1, 0,
                                0, -1, -1, 0, 0, 1],
                         'x2': [0, 1, 1, 0, -1, -1, 1, -1, 1, 0, 1, 1, 1, 0,
                                0, -1, -1, 0, 0, 1]})
    data['x1_pre'] = get_pre_val_Fcond(data, 'x1', 'x2', lambda x: x != 1)
    data:
            x1  x2  x1_pre
        0    0   0     NaN
        1    1   1     0.0
        2    1   1     0.0
        3    0   0     0.0
        4   -1  -1     0.0
        5   -1  -1    -1.0
        6    2   1    -1.0
        7   -1  -1    -1.0
        8    1   1    -1.0
        9    0   0    -1.0
        10   1   1     0.0
        11   1   1     0.0
        12   1   1     0.0
        13   0   0     0.0
        14   0   0     0.0
        15  -1  -1     0.0
        16  -1  -1    -1.0
        17   0   0    -1.0
        18   0   0     0.0
        19   1   1     0.0
    '''
    
    df = data[[col_val, col_Fcond]].copy()    
    # 为了避免计算过程中临时产生的列名与原始列名混淆，对列重新命名
    col_val, col_Fcond = ['col_val', 'col_Fcond']
    df.columns = [col_val, col_Fcond]
    
    df['Fok'] = df[col_Fcond].apply(lambda x: Fcond(x)).astype(int)
    df['val_pre'] = df[df['Fok'] == 1][col_val]
    df['val_pre'] = df['val_pre'].shift(1).fillna(method='ffill')
    
    return df['val_pre']


def gap_count(series, Fcond, via_pd=True):
    '''    
    计算series(pd.Series)中当前行距离上一个满足Fcond函数指定条件记录的行数
    Fcond为指定条件的函数，Fcond(x)返回结果只能为True或False，
    若via_pd为False，则使用循环迭代，若via_pd为True，则使用pd
    返回结果为pd.Series
    
    Examples
    --------
    df = pd.DataFrame([0, 1, 1, 0, 0, 1, 1, 1], columns=['series'])
    Fcond = lambda x: True if x == 1 else False
    df['gap1'] = gap_count(df['series'], Fcond, True)
    df:
           series  gap1
        0       0     0
        1       1     0
        2       1     1
        3       0     1
        4       0     2
        5       1     3
        6       1     1
        7       1     1
    df['gap0'] = gap_count(df['series'], lambda x: x != 1, False)
    df:
           series  gap1  gap0
        0       0     0     0
        1       1     0     1
        2       1     1     2
        3       0     1     3
        4       0     2     1
        5       1     3     1
        6       1     1     2
        7       1     1     3
    '''
    
    col = 'series'
    series.name = col    
    df = pd.DataFrame(series)
    
    # 当series.index存在重复值时为避免报错，因此先重置index最后再还原
    ori_index = df.index
    df.index = range(0, df.shape[0])
    
    if via_pd:        
        df['idx'] = range(0, df.shape[0])
        df['idx_pre'] = get_pre_val_Fcond(df, 'idx', col, Fcond)
        df['gap'] = (df['idx'] - df['idx_pre']).fillna(0).astype(int)
        
        df.index = ori_index
        
        return df['gap']
    
    else:
        df['count'] = con_count(series, lambda x: not Fcond(x), via_pd=via_pd)
        
        df['gap'] = df['count']
        k0 = 0
        while k0 < df.shape[0] and not Fcond(df.loc[df.index[k0], col]):
            df.loc[df.index[k0], 'gap'] = 0
            k0 += 1
            
        for k1 in range(k0+1, df.shape[0]):
            if Fcond(df.loc[df.index[k1], col]):
                df.loc[df.index[k1], 'gap'] = \
                                        df.loc[df.index[k1-1], 'count'] + 1
                                        
        df.index = ori_index
            
        return df['gap']
    
    
def count_between_gap(data, col_gap, col_count, gapFcond, countFcond,
                      count_now_gap=True, count_now=True, via_pd=True):
    '''    
    计算data中当前行与上一个满足gapFcond指定条件的行之间，
    满足countFcond函数指定条件的记录数
    其中函数gapFond作用于col_gap列，countFcond作用于col_count列
    count_now_gap设置满足gapFcond的行是否参与计数
    count_now设置当当前行满足countFcond时，从当前行开始对其计数还是从下一行开始对其计数
    注：当前行若满足同时满足gapFcond和countFcond，对其计数的行不会为下一行
    （即要么不计数，要么在当前行对其计数）
    若via_pd为True，则调用count_between_gap_pd实现，否则用count_between_gap_iter
    返回结果为pd.Series
    
    Examples
    --------
    data = pd.DataFrame({'to_gap': [0, 1, 1, 0, -1, -1, 2, -1, 1, 0, 1, 1,
                                    1, 0, 0, -1, -1, 0, 0, 1],
                         'to_count': [0, 1, 1, 0, -1, -1, 1, -1, 1, 0, 1,
                                      1, 1, 0, 0, -1, -1, 0, 0, 1]})
    data['gap_count'] = count_between_gap(data, 'to_gap', 'to_count',
                                    lambda x: x == -1, lambda x: x == 1,
                                    count_now_gap=False, count_now=False)
    data:
            to_gap  to_count  gap_count
        0        0         0          0
        1        1         1          0
        2        1         1          0
        3        0         0          0
        4       -1        -1          0
        5       -1        -1          0
        6        2         1          1
        7       -1        -1          0
        8        1         1          1
        9        0         0          1
        10       1         1          2
        11       1         1          3
        12       1         1          4
        13       0         0          4
        14       0         0          4
        15      -1        -1          0
        16      -1        -1          0
        17       0         0          0
        18       0         0          0
        19       1         1          1
            
    data = pd.DataFrame({'to_gap': [0, 1, 1, 0, -1, -1, 2, -1, 1, 0, 1, 1,
                                    1, 0, 0, -1, -1, 0, 0, 1, -1, 1],
                         'to_count': [0, 1, 1, 0, -1, -1, 1, -1, 1, 0, 1,
                                      1, 1, 0, 0, -1, 1, 0, 1, 1, 1, -1]})
    data['gap_count'] = count_between_gap(data, 'to_gap', 'to_count',
                                      lambda x: x == -1, lambda x: x == 1,
                                      count_now_gap=True, count_now=True)
    data:
            to_gap  to_count  gap_count
        0        0         0          0
        1        1         1          0
        2        1         1          0
        3        0         0          0
        4       -1        -1          0
        5       -1        -1          0
        6        2         1          1
        7       -1        -1          1
        8        1         1          1
        9        0         0          1
        10       1         1          2
        11       1         1          3
        12       1         1          4
        13       0         0          4
        14       0         0          4
        15      -1        -1          4
        16      -1         1          1
        17       0         0          0
        18       0         1          1
        19       1         1          2
        20      -1         1          3
        21       1        -1          0
            
    data = pd.DataFrame({'to_gap': [0, 1, 1, 0, -1, -1, 2, -1, 1, 0, 1, 1,
                                    1, 0, 0, -1, -1, 0, 0, 1, -1, 1],
                         'to_count': [0, -1, -1, 0, -1, -1, 1, -1, 1, 0, 1, 1,
                                      1, 0, 0, -1, -1, 0, -1, 1, 1, -1]})
    data['gap_count'] = count_between_gap(data, 'to_gap', 'to_count',
                                      lambda x: x == -1, lambda x: x == 1,
                                      count_now_gap=True, count_now=False)
    data:
            to_gap  to_count  gap_count
        0        0         0          0
        1        1        -1          0
        2        1        -1          0
        3        0         0          0
        4       -1        -1          0
        5       -1        -1          0
        6        2         1          0
        7       -1        -1          1
        8        1         1          0
        9        0         0          1
        10       1         1          1
        11       1         1          2
        12       1         1          3
        13       0         0          4
        14       0         0          4
        15      -1        -1          4
        16      -1        -1          0
        17       0         0          0
        18       0        -1          0
        19       1         1          0
        20      -1         1          1
        21       1        -1          0
            
    data = pd.DataFrame({'to_gap': [0, 1, 1, 0, -1, -1, 2, -1, 1, 0, 1, 1, 1,
                                    0, 0, -1, -1, 0, 0, 1, -1, 1],
                         'to_count': [0, -1, -1, 0, -1, -1, 1, -1, 1, 0, 1, 1,
                                      1, 0, 0, -1, -1, 0, -1, 1, 1, -1]})
    data['gap_count'] = count_between_gap(data, 'to_gap', 'to_count',
                                      lambda x: x == -1, lambda x: x == 1,
                                      count_now_gap=False, count_now=True)
    data:
            to_gap  to_count  gap_count
        0        0         0          0
        1        1        -1          0
        2        1        -1          0
        3        0         0          0
        4       -1        -1          0
        5       -1        -1          0
        6        2         1          1
        7       -1        -1          0
        8        1         1          1
        9        0         0          1
        10       1         1          2
        11       1         1          3
        12       1         1          4
        13       0         0          4
        14       0         0          4
        15      -1        -1          0
        16      -1        -1          0
        17       0         0          0
        18       0        -1          0
        19       1         1          1
        20      -1         1          0
        21       1        -1          0
    '''
    
    if via_pd:
        return count_between_gap_pd(data, col_gap, col_count, gapFcond,
                                    countFcond, count_now_gap=count_now_gap,
                                    count_now=count_now)
    else:
        return count_between_gap_iter(data, col_gap, col_count, gapFcond,
                                      countFcond, count_now_gap=count_now_gap,
                                      count_now=count_now)
    
    
def count_between_gap_pd(data, col_gap, col_count, gapFcond, countFcond,
                         count_now_gap=True, count_now=True):
    '''参数和功能说明见count_between_gap函数'''
    
    df = data[[col_gap, col_count]].copy()
    # 为了避免计算过程中临时产生的列名与原始列名混淆，对列重新命名
    col_gap, col_count = ['col_gap', 'col_count']
    df.columns = [col_gap, col_count]    
    
    df['gap0'] = df[col_gap].apply(lambda x: not gapFcond(x)).astype(int)
    df['count1'] = df[col_count].apply(lambda x: countFcond(x)).astype(int)
    df['gap_count'] = df[df['gap0'] == 1]['count1'].cumsum()
    df['gap_cut'] = df['gap0'].diff().shift(-1)
    df['gap_cut'] = df['gap_cut'].apply(lambda x: 1 if x == -1 else np.nan)
    df['tmp'] = (df['gap_count'] * df['gap_cut']).shift(1)
    df['tmp'] = df['tmp'].fillna(method='ffill')
    df['gap_count'] = df['gap_count'] - df['tmp']
    
    if count_now_gap:
        df['pre_gap0'] = df['gap0'].shift(1)
        df['tmp'] = df['gap_count'].shift()        
        df['tmp'] = df[df['gap0'] == 0]['tmp'] 
        
        df['gap_count1'] = df['gap_count'].fillna(0)
        df['gap_count2'] = df['tmp'].fillna(0) + df['count1'] * (1-df['gap0'])
        df['gap_count'] = df['gap_count1'] + df['gap_count2']
        
    if not count_now:
        df['gap_count'] = df['gap_count'].shift(1)
        if not count_now_gap:
            df['gap_count'] = df['gap0'] * df['gap_count']
        else:
            df['gap_count'] = df['pre_gap0'] * df['gap_count']
        
    df['gap_count'] = df['gap_count'].fillna(0).astype(int)
        
    return df['gap_count']


def count_between_gap_iter(data, col_gap, col_count, gapFcond, countFcond,
                           count_now_gap=True, count_now=True):
    '''参数和功能说明见count_between_gap函数'''
    
    df = data[[col_gap, col_count]].copy()
    # 为了避免计算过程中临时产生的列名与原始列名混淆，对列重新命名
    col_gap, col_count = ['col_gap', 'col_count']
    df.columns = [col_gap, col_count]
    
    # 当data.index存在重复值时为避免报错，因此先重置index最后再还原
    ori_index = df.index
    df.index = range(0, df.shape[0])
    
    df['gap_count'] = 0
    
    k = 0
    while k < df.shape[0]:
        if gapFcond(df.loc[df.index[k], col_gap]):
            k += 1
            gap_count = 0
            while k < df.shape[0] and \
                                  not gapFcond(df.loc[df.index[k], col_gap]):
                if countFcond(df.loc[df.index[k], col_count]):
                    gap_count += 1
                df.loc[df.index[k], 'gap_count'] = gap_count
                k += 1
        else:
            k += 1
            
    if count_now_gap:
        k = 1
        while k < df.shape[0]:
            if gapFcond(df.loc[df.index[k], col_gap]):
                if not gapFcond(df.loc[df.index[k-1], col_gap]):
                    if countFcond(df.loc[df.index[k], col_count]):
                        df.loc[df.index[k], 'gap_count'] = \
                                        df.loc[df.index[k-1], 'gap_count'] + 1
                        k += 1 
                    else:
                        df.loc[df.index[k], 'gap_count'] = \
                                            df.loc[df.index[k-1], 'gap_count']
                        k += 1
                else:
                    if countFcond(df.loc[df.index[k], col_count]):
                        df.loc[df.index[k], 'gap_count'] = 1
                        k += 1
                    else:
                        k += 1
            else:
                k += 1
                
    if not count_now:
        df['gap_count_pre'] = df['gap_count'].copy()
        if not count_now_gap:
            for k in range(1, df.shape[0]):
                if gapFcond(df.loc[df.index[k], col_gap]):
                    df.loc[df.index[k], 'gap_count'] = 0
                else:
                    df.loc[df.index[k], 'gap_count'] = \
                                        df.loc[df.index[k-1], 'gap_count_pre']
        else:
            for k in range(1, df.shape[0]):                
                if gapFcond(df.loc[df.index[k-1], col_gap]):
                    df.loc[df.index[k], 'gap_count'] = 0
                else:
                    df.loc[df.index[k], 'gap_count'] = \
                                        df.loc[df.index[k-1], 'gap_count_pre']
        df.drop('gap_count_pre', axis=1, inplace=True)
                
    k0 = 0
    while k0 < df.shape[0] and not gapFcond(df.loc[df.index[k0], col_gap]):
        df.loc[df.index[k0], 'gap_count'] = 0
        k0 += 1
    df.loc[df.index[k0], 'gap_count'] = 0
    
    df.index = ori_index
                                       
    return df['gap_count']


def replace_repeat_iter(series, val, val0, gap=None):
    '''
    替换重复出现的值
    series中若步长为gap的范围内出现多个val值，则只保留第一条记录，后面的替换为val0
    若gap为None，则将连续出现的val值只保留第一个，其余替换为val0(这里连续出现val是指
    不出现除了val和val0之外的其他值)
    返回结果为替换之后的pd.Series
    
    Examples
    --------
    data = pd.DataFrame([0, 1, 1, 0, -1, -1, 2, -1, 1, 0, 1, 1, 1, 0, 0,
                         -1, -1, 0, 0, 1], columns=['test'])
    data['test_rep'] = replace_repeat_iter(data['test'], 1, 0, gap=None)
    data:
            test  test_rep
        0      0         0
        1      1         1
        2      1         0
        3      0         0
        4     -1        -1
        5     -1        -1
        6      2         2
        7     -1        -1
        8      1         1
        9      0         0
        10     1         0
        11     1         0
        12     1         0
        13     0         0
        14     0         0
        15    -1        -1
        16    -1        -1
        17     0         0
        18     0         0
        19     1         1
    '''
    
    col = series.name
    df = pd.DataFrame({col: series})
    
    if gap is not None and (gap > df.shape[0] or gap < 1):
        raise ValueError('gap取值范围必须为1到df.shape[0]之间！')
    gap = None if gap == 1 else gap
    
    # 当series.index存在重复值时为避免报错，因此先重置index最后再还原
    ori_index = df.index
    df.index = range(0, df.shape[0])
    
    k = 0
    while k < df.shape[0]:
        if df.loc[df.index[k], col] == val:
            k1 = k + 1
            
            if gap is None:
                while k1 < df.shape[0] and \
                                    df.loc[df.index[k1], col] in [val, val0]:
                    if df.loc[df.index[k1], col] == val:
                        df.loc[df.index[k1], col] = val0
                    k1 += 1
            else:
                while k1 < min(k+gap, df.shape[0]):
                    if df.loc[df.index[k1], col] == val:
                        df.loc[df.index[k1], col] = val0
                    k1 += 1
            k =  k1
            
        else:
            k += 1
            
    df.index = ori_index
            
    return df[col]


def replace_repeat_pd(series, val, val0):
    '''
    series连续出现的重复val值保留第一个，其余替换为val0（用pd，不用循环迭代）
    参数和意义同replace_repeat_iter函数，返回结果为替换之后的pd.Series
    '''
    
    col = series.name
    df = pd.DataFrame({col: series})
    # 为了避免计算过程中临时产生的列名与原始列名混淆，对列重新命名
    col_ori = col
    col = 'series'
    df.columns = [col]
    
    # 当series.index存在重复值时为避免报错，因此先重置index最后再还原
    ori_index = df.index
    df.index = range(0, df.shape[0])
    
    df['gap1'] = df[col].apply(lambda x: x not in [val, val0]).astype(int)
    df['is_val'] = df[col].apply(lambda x: x == val).astype(int)
    df['val_or_gap'] = df['gap1'] + df['is_val']
    df['pre_gap'] = df[df['val_or_gap'] == 1]['gap1'].shift(1)
    df['pre_gap'] = df['pre_gap'].fillna(method='ffill')
    k = 0
    while k < df.shape[0] and df.loc[df.index[k], 'is_val'] != 1:
        k += 1
    if k < df.shape[0]:
        df.loc[df.index[k], 'pre_gap'] = 1
    df['pre_gap'] = df['pre_gap'].fillna(0).astype(int)
    df['keep1'] = (df['is_val'] + df['pre_gap']).map({0: 0, 1: 0, 2: 1})
    df['to_rplc'] = (df['keep1'] + df['is_val']).map({2: 0, 1: 1, 0: 0})
    df[col] = df[[col, 'to_rplc']].apply(lambda x: 
                            val0 if x['to_rplc'] == 1 else x[col], axis=1)
        
    df.rename(columns={col: col_ori}, inplace=True)
    df.index = ori_index
    
    return df[col_ori]


def replace_repeat_F_iter(series, valF, val0F, gap=None):
    '''
    series中连续出现的满足valF函数的重复值保留第一个，其余替换为val0F函数的值
    与replace_repeat作用一样，只不过把val和val0由指定值换成了由函数生成值，
    valF函数用于判断连续条件，其返回值只能是True或False，val0F函数用于生成替换的新值    
    series中若步长为gap的范围内出现多个满足valF函数为True的值，则只保留第一条记录，
    后面的替换为函数val0F的值
    若gap为None，则将连续出现的满足valF函数为True的值只保留第一个，
    其余替换为函数val0F的值(这里连续出现是指不出现除了满足valF为True和等于val0F函数值
    之外的其他值)
    返回结果为替换之后的pd.Series
    
    Examples
    --------
    data = pd.DataFrame([0, 1, 1, 0, -1, -1, 2, -1, 1, 0, 1, 1, 1, 0, 0,
                         -1, -1, 0, 0, 1], columns=['y'])
    data['y_rep'] = replace_repeat_F_iter(data['y'], lambda x: x < 1,
                                          lambda x: 3, gap=None)
    data:
            y  y_rep
        0   0      0
        1   1      1
        2   1      1
        3   0      0
        4  -1      3
        5  -1      3
        6   2      2
        7  -1     -1
        8   1      1
        9   0      0
        10  1      1
        11  1      1
        12  1      1
        13  0      0
        14  0      3
        15 -1      3
        16 -1      3
        17  0      3
        18  0      3
        19  1      1
    '''
    
    col = series.name
    df = pd.DataFrame({col: series})
    
    if gap is not None and (gap > df.shape[0] or gap < 1):
        raise ValueError('gap取值范围必须为1到df.shape[0]之间！')
    gap = None if gap == 1 else gap
    
    # 当series.index存在重复值时为避免报错，因此先重置index最后再还原
    ori_index = df.index
    df.index = range(0, df.shape[0])
    
    k = 0
    while k < df.shape[0]:
        if valF(df.loc[df.index[k], col]):
            k1 = k + 1
            
            if gap is None:
                while k1 < df.shape[0] and (valF(df.loc[df.index[k1], col]) \
            or df.loc[df.index[k1], col] == val0F(df.loc[df.index[k1], col])):
                    if valF(df.loc[df.index[k1], col]):
                        df.loc[df.index[k1], col] = \
                                              val0F(df.loc[df.index[k1], col])
                    k1 += 1
            else:
                while k1 < min(k+gap, df.shape[0]):
                    if valF(df.loc[df.index[k1], col]):
                        df.loc[df.index[k1], col] = \
                                              val0F(df.loc[df.index[k1], col])
                    k1 += 1
            k =  k1
            
        else:
            k += 1
            
    df.index = ori_index
            
    return df[col]


def replace_repeat_F_pd(series, valF, val0F):
    '''
    series中连续出现的满足valF函数的重复值保留第一个，其余替换为val0F函数的值
    （用pd，不用循环迭代）
    参数和意义同replace_repeat_F_iter函数，返回结果为替换之后的pd.Series
    '''
    
    col = series.name
    df = pd.DataFrame({col: series})
    # 为了避免计算过程中临时产生的列名与原始列名混淆，对列重新命名
    col_ori = col
    col = 'series'
    df.columns = [col]
    
    # 当series.index存在重复值时为避免报错，因此先重置index最后再还原
    ori_index = df.index
    df.index = range(0, df.shape[0])
    
    df['gap1'] = df[col].apply(lambda x: 
                               not valF(x) and x != val0F(x)).astype(int)
    df['is_val'] = df[col].apply(lambda x: valF(x)).astype(int)
    df['val_or_gap'] = df['gap1'] + df['is_val']
    df['pre_gap'] = df[df['val_or_gap'] == 1]['gap1'].shift(1)
    df['pre_gap'] = df['pre_gap'].fillna(method='ffill')
    k = 0
    while k < df.shape[0] and df.loc[df.index[k], 'is_val'] != 1:
        k += 1
    if k < df.shape[0]:
        df.loc[df.index[k], 'pre_gap'] = 1
    df['pre_gap'] = df['pre_gap'].fillna(0).astype(int)
    df['keep1'] = (df['is_val'] + df['pre_gap']).map({0: 0, 1: 0, 2: 1})
    df['to_rplc'] = (df['keep1'] + df['is_val']).map({2: 0, 1: 1, 0: 0})
    df[col] = df[[col, 'to_rplc']].apply(lambda x: 
                    val0F(x[col]) if x['to_rplc'] == 1 else x[col], axis=1)
        
    df.rename(columns={col: col_ori}, inplace=True)
    df.index = ori_index
    
    return df[col_ori]


def val_gap_cond(data, col_Fval, col_Fcond, Fcond, Fval,
                 to_cal_col=None, Fto_cal=None, Vnan=np.nan,
                 contain_1st=False):
    '''
    计算从上一个满足Fcond函数的记录到当前行，col_Fval列记录的Fval函数值
    Fcond作用于col_Fcond列，Fcond(x)返回True或False，x为单个值
    Fval函数作用于col_Fval列，Fval(x)返回单个值，x为np.array或pd.Series或列表等
    Fto_cal作用于to_cal_col列，只有当前行Fto_cal值为True时才进行Fval计算，
    否则返回结果中当前行值设置为Vnan
    contain_1st设置Fval函数计算时是否将上一个满足Fcond的行也纳入计算
    
    Examples
    --------
    data = pd.DataFrame({'val': [1, 2, 5, 3, 1, 7 ,9],
                         'sig': [1, 1, -1, 1, 1, -1, 1]})
    data['val_pre1'] = val_gap_cond(data, 'val', 'sig',
                                    lambda x: x == -1, lambda x: max(x))
    data:
           val  sig  val_pre1
        0    1    1       NaN
        1    2    1       NaN
        2    5   -1       NaN
        3    3    1       3.0
        4    1    1       3.0
        5    7   -1       7.0
        6    9    1       9.0
    '''
    
    if to_cal_col is None and Fto_cal is None:
        df = data[[col_Fval, col_Fcond]].copy()
        # 为了避免计算过程中临时产生的列名与原始列名混淆，对列重新命名
        col_Fval, col_Fcond = ['col_Fval', 'col_Fcond']
        df.columns = [col_Fval, col_Fcond]
    elif to_cal_col is not None and Fto_cal is not None:
        df = data[[col_Fval, col_Fcond, to_cal_col]].copy()
        # 为了避免计算过程中临时产生的列名与原始列名混淆，对列重新命名
        col_Fval, col_Fcond, to_cal_col = ['col_Fval', 'col_Fcond',
                                                               'to_cal_col']
        df.columns = [col_Fval, col_Fcond, to_cal_col]
    
    df['idx'] = range(0, df.shape[0])
    df['pre_idx'] = get_pre_val_Fcond(df, 'idx', col_Fcond, Fcond)
    
    if to_cal_col is None and Fto_cal is None:
        if not contain_1st:
            df['gap_val'] = df[['pre_idx', 'idx', col_Fval]].apply(lambda x:
               Fval(df[col_Fval].iloc[int(x['pre_idx']+1): int(x['idx']+1)]) \
               if not isnull(x['pre_idx']) else Vnan, axis=1)
        else:
            df['gap_val'] = df[['pre_idx', 'idx', col_Fval]].apply(lambda x:
               Fval(df[col_Fval].iloc[int(x['pre_idx']): int(x['idx']+1)]) \
               if not isnull(x['pre_idx']) else Vnan, axis=1)
    elif to_cal_col is not None and Fto_cal is not None:
        if not contain_1st:
            df['gap_val'] = df[['pre_idx', 'idx', col_Fval,
                                to_cal_col]].apply(
    lambda x: Fval(df[col_Fval].iloc[int(x['pre_idx']+1): int(x['idx']+1)]) \
              if not isnull(x['pre_idx']) and Fto_cal(x[to_cal_col]) else \
              Vnan, axis=1)
        else:
            df['gap_val'] = df[['pre_idx', 'idx', col_Fval,
                                to_cal_col]].apply(
    lambda x: Fval(df[col_Fval].iloc[int(x['pre_idx']): int(x['idx']+1)]) \
              if not isnull(x['pre_idx']) and Fto_cal(x[to_cal_col]) else \
              Vnan, axis=1)
        
    return df['gap_val']


def filter_by_FPrePost_series(series, FPrePost, Fignore=lambda x: isnull(x),
                              Vnan=np.nan):
    '''
    对series调用filter_by_FPrePost函数，其中满足Fignore函数的值不参与
    series中被过滤的值在返回结果中用Vana替换（不参与的值保持不变）
    
    Examples
    --------
    series = pd.Series([1, 2, 3, 4, 1, 1, 2, 3, 6])
    FPrePost = lambda x, y: (y-x) >= 2
    filter_by_FPrePost_series(series, FPrePost)
        0    1.0
        1    NaN
        2    3.0
        3    NaN
        4    NaN
        5    NaN
        6    NaN
        7    NaN
        8    6.0
    series = pd.Series([1, 2, 0, 3, 0, 4, 0, 1, 0, 0, 1, 2, 3, 6],
                       index=range(14, 0, -1))
    filter_by_FPrePost_series(series, FPrePost, lambda x: x == 0)
    '''
    
    l = [[k, series.iloc[k]] for k in range(0, len(series)) \
                                             if not Fignore(series.iloc[k])]
    lnew = filter_by_FPrePost(l, lambda x, y: FPrePost(x[1], y[1]))
    
    i_l = [k for k, v in l]
    i_lnew = [k for k, v in lnew]    
    idxs_ignore = [_ for _ in i_l if _ not in i_lnew]
    
    seriesNew = series.copy()
    for k in idxs_ignore:
        seriesNew.iloc[k] = Vnan
        
    return seriesNew
        

def filter_by_FPrePost(l, FPrePost):
    '''
    对l（list）进行过滤，过滤后的lnew（list）前后相邻两个值满足：
        FPrePost(lnew[i], lnew[i+1]) = True
    过滤过程为：将l的第一个元素作为起点，找到其后第一个满足FPrePost函数的元素，
        再以该元素为起点往后寻找...
    
    Parameters
    ----------
    l(list): 待过滤列表
    FPrePost: 过滤函数，接收两个参数，返回值为True或False
    
    Returns
    -------
    lnew(list): 过滤后的列表
    
    Examples
    --------
    l = [1, 2, 3, 4, 1, 1, 2, 3, 6]
    FPrePost = lambda x, y: (y-x) >= 2
    filter_by_FPrePost(l, FPrePost)
    >>> [1, 3, 6]
    filter_by_FPrePost(l, lambda x, y: y == x+1)
    >>> [1, 2, 3, 4]
    '''
    
    if len(l) == 0:
        return l
    
    lnew = [l[0]]
    idx_pre, idx_post = 0, 1
    while idx_post < len(l):
        vpre = l[idx_pre]
        idx_post = idx_pre + 1
        
        while idx_post < len(l):
            vpost = l[idx_post]
            
            if not FPrePost(vpre, vpost):
                idx_post += 1
            else:
                lnew.append(vpost)
                idx_pre = idx_post
                break
            
    return lnew


def min_com_multer(l):
    '''求一列数l的最小公倍数，支持负数和浮点数'''
    l_max = max(l)
    mcm = l_max
    while any([mcm % x != 0 for x in l]):
        mcm += l_max
    return mcm


def max_com_divisor(l):
    '''
    求一列数l的最大公约数，只支持正整数
    '''
    
    def isint(x):
        '''判断x是否为整数'''
        tmp = str(x).split('.')
        if len(tmp) == 1 or all([x == '0' for x in tmp[1]]):
            return True
        return False
    
    if any([x < 1 or not isint(x) for x in l]):
        raise ValueError('只支持正整数！')
    
    l_min = min(l)
    mcd = l_min
    while any([x % mcd != 0 for x in l]):
        mcd -= 1
        
    return mcd


def mcd2_tad(a, b):
    '''辗转相除法求a和b的最大公约数，a、b为正数，为小数时由于精度问题会不正确'''
    if a < b:
        a, b = b, a # a存放较大值，b存放较小值
    if a % b == 0:
        return b
    else:
        return mcd2_tad(b, a % b)


def max_com_divisor_tad(l):
    '''
    用辗转相除法求一列数l的最大公约数，l元素均为正数，为小数时由于精度问题会不正确
    https://blog.csdn.net/weixin_45069761/article/details/107954905
    '''
    # g = l[0]
    # for i in range(1, len(l)):
    #     g = mcd2_tad(g, l[i])
    # return g
    
    return reduce(lambda x, y: mcd2_tad(x, y), l)
    

def isnull(x):
    '''判断x是否为无效值（None或nan）'''
    if x is None:
        return True
    if x is np.nan:
        return True
    try:
        if x != x:
            return True
    except:
        pass
    return False


def x_div_y(x, y, v_x0=None, v_y0=0, v_xy0=1):
    '''
    x除以y，
    v_xy0为当x和y同时为0时的返回值，
    v_y0为当y等于0时的返回值，
    v_x0为当x等于0时的返回值
    '''
    if x == 0 and y == 0:
        return v_xy0
    if x != 0 and y == 0:
        return v_y0
    if x == 0 and y != 0:
        return 0 if v_x0 is None else v_x0
    return x / y
    
    
def merge_df(df_left, df_right, same_keep='left', **kwargs):
    '''
    pd.merge，相同列名时去除重复
    same_keep可选['left', 'right']设置相同列保留左边df还是右边df
    **kwargs接收pd.merge接受的其他参数
    '''
    same_cols = [x for x in df_left.columns if x in df_right.columns]
    if len(same_cols) > 0:
        if 'on' in kwargs:
            if isinstance(kwargs['on'], list):
                same_cols = [x for x in same_cols if x not in kwargs['on']]
            elif isinstance(kwargs['on'], str):
                same_cols = [x for x in same_cols if x != kwargs['on']]
            else:
                raise ValueError('on参数只接受list或str！')
        if same_keep == 'left':
            df_right = df_right.drop(same_cols, axis=1)
        elif same_keep == 'right':
            df_left = df_left.drop(same_cols, axis=1)
        else:
            raise ValueError('same_keep参数只接受`left`或`right`！')
    return pd.merge(df_left, df_right, **kwargs)


def cut_df_by_con_val(df, by_col):
    '''
    根据by_col列的值，将df切分为多个子集列表
    切分依据：by_col列中值连续相等的记录被划分到一个子集中
    Examples
    --------
        df = pd.DataFrame({'val':range(0,10),
                           'by_col': ['a']*3+['b']*2+['c']*1+['a']*3+['d']*1})
        cut_df_by_con_val(df, 'by_col')
            [   val by_col
             0    0      a
             1    1      a
             2    2      a,
                val by_col
             3    3      b
             4    4      b,
                val by_col
             5    5      c,
                val by_col
             6    6      a
             7    7      a
             8    8      a,
                val by_col
             9    9      d]
    '''
    
    df_ = df.reset_index(drop=True) # 当df.index存在重复值时可能报错，故index去重
    
    sub_dfs= []
    k = 0
    while k < df_.shape[0]:
        k1 = k + 1
        while k1 < df_.shape[0] and \
              df_.loc[df_.index[k1], by_col] == df_.loc[df.index[k], by_col]:
            k1 += 1
        sub_dfs.append(df_.iloc[k:k1, :])
        k = k1
        
    return sub_dfs


def get_con_start_end(series, Fcond):
    '''
    找出series中值连续满足Fcond函数的分段起止位置
    
    Example
    -------
    series = pd.Series([0, 1, 1, 0, 1, 1, 0, -1, -1, 0, 0, -1, 1, 1, 1, 1, 0,
                        -1])
    start_ends = get_con_start_end(series, lambda x: x == -1)
    start_ends:
        [[7, 8], [11, 11], [17, 17]]
    start_ends = get_con_start_end(series, lambda x: x == 1)
    start_ends:
        [[1, 2], [4, 5], [12, 15]]
    '''
    
    start_ends = []
    # df['start'] = 0
    # df['end'] = 0
    start = 0
    N = len(series)
    while start < N:
        if Fcond(series.iloc[start]):
            end = start
            while end < N and Fcond(series.iloc[end]):
                end += 1
            start_ends.append([start, end-1])
            # df.loc[df.index[start], 'start'] = 1
            # df.loc[df.index[end-1], 'end'] = 1
            start = end + 1
        else:
            start += 1
            
    return start_ends
    

def check_exist_data(df, x_list, cols=None):
    '''
    依据指定的cols列检查df中是否已经存在x_list中的记录
    
    Examples
    --------
    df = pd.DataFrame([['1', 2, 3.1, ], ['3', 4, 5.1], ['5', 6, 7.1]],
                      columns=['a', 'b', 'c'])
    x_list, cols = [[3, 4], ['3', 4]], ['a', 'b']
    check_exist_data(df, x_list, cols=cols):
        [False, True]
    check_exist_data(df, [['1', 3.1], ['3', 5.1]], ['a', 'c']):
        [True, True]
    '''
    
    if not isnull(cols):
        df_ = df.reindex(columns=cols)
    else:
        df_ = df.copy()
    data = df_.to_dict('split')['data']
    return [x in data for x in x_list]


def check_l_in_l0(l, l0):
    '''
    判断l（list）中的值是否都是l0（list）中的元素
    
    Example
    -------
    l = [1, 2, 3, -1, 0]
    l0 = [0, 1, -1]
    check_l_in_l0(l, l0)
    >>> False
    
    l = [1, 1, 0, -1, -1, 0, 0]
    l0 = [0, 1, -1]
    check_l_in_l0(l, l0)
    >>> True
    '''
    l_ = set(l)
    l0_ = set(l0)
    return len(l_-l0_) == 0
    
    
if __name__ == '__main__':
    from utils_hoo import load_csv
    from utils_hoo.utils_fin.utils_fin import CCI
    from utils_hoo.utils_plot.plot_Common import plot_Series
    
    # 50ETF日线行情------------------------------------------------------------
    fpath = './test/510050_daily_pre_fq.csv'
    data = load_csv(fpath)
    data.set_index('date', drop=False, inplace=True)
    
    data['cci'] = CCI(data)
    data['cci_100'] = data['cci'].apply(lambda x: 1 if x > 100 else \
                                                    (-1 if x < -100 else 0))
        
    plot_Series(data.iloc[-200:, :], {'close': ('.-k', False)},
                cols_styl_low_left={'cci': ('.-c', False)},
                cols_to_label_info={'cci':
                                [['cci_100', (-1, 1), ('r^', 'bv'), False]]},
                xparls_info={'cci': [(100, 'r', '-', 1.3),
                                     (-100, 'r', '-', 1.3)]},
                figsize=(8, 7), grids=True)
        
    start_ends_1 = get_con_start_end(data['cci_100'], lambda x: x == -1)
    start_ends1 = get_con_start_end(data['cci_100'], lambda x: x == 1)
    data['cci_100_'] = 0
    for start, end in start_ends_1:
        if end+1 < data.shape[0]:
            data.loc[data.index[end+1], 'cci_100_'] = -1
    for start, end in start_ends1:
        if end+1 < data.shape[0]:
            data.loc[data.index[end+1], 'cci_100_'] = 1
            
    plot_Series(data.iloc[-200:, :], {'close': ('.-k', False)},
                cols_styl_low_left={'cci': ('.-c', False)},
                cols_to_label_info={'cci':
                                [['cci_100_', (-1, 1), ('r^', 'bv'), False]],
                                    'close':
                                [['cci_100_', (-1, 1), ('r^', 'bv'), False]]},
                xparls_info={'cci': [(100, 'r', '-', 1.3),
                                     (-100, 'r', '-', 1.3)]},
                figsize=(8, 7), grids=True)
    
    
    
    
    
    
    
    