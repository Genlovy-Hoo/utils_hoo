# -*- coding: utf-8 -*-

import time
import math
import random
import numpy as np
import pandas as pd
from test_funcs import TestFuncs
from utils_Heuristic import FuncOpterInfo, rand_init
from utils_hoo.utils_plot.plot_Common import plot_Series
from utils_hoo.utils_general import simple_logger, isnull
from utils_hoo.utils_logging.logger_general import get_logger
from utils_hoo.utils_logging.logger_utils import close_log_file


def Levy(dim, beta=1.5, alpha=0.01):
    '''
    Levy飞行
    参考：
    sigma、u、v、s意义见https://www.jianshu.com/p/4f6e02fc8396
    '''
    
    tmp1 = math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
    tmp2 = math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma = (tmp1 / tmp2) ** (1 / beta)
    
    u = np.random.randn(dim) * sigma # 正态分布 u ~ N(0, sigma^2)
    v = np.random.randn(dim) # 标准正态分布 v ~ N(0, 1)
    
    step = alpha * u / (abs(v) ** (1 / beta))
    
    return step


def HHO(objf, func_opter_parms):
    '''
    todo: 目前仅考虑自变量连续实数情况，后面可增加自变量为离散的情况
    
    哈里斯鹰优化算法(Harris Hawks Optimizer) HHO algorithm
    
    Parameters
    ----------
    objf: 目标函数，须事先转化为求极小值问题
    func_opter_parms: FuncOpterInfo类，须设置parms_func、parms_opter、parms_log
    parms_func: 目标函数参数信息dict，key须包含
        x_lb: 自变量每个维度取值下界，list或数值，为list时长度应等于dim
        x_ub: 自变量每个维度取值上界，list或数值，为list时长度应等于dim
        dim: 自变量维度数
        
        kwargs: 目标函数接收的其它参数
    parms_opter: 优化函数参数信息dict，key须包含
        PopSize: 群体数量（每轮迭代的样本数量）
        Niter: 最大迭代寻优次数
        
        alpha, beta: Levy飞行参数
    parms_log: 日志参数信息dict，key须包含
        logger: 日志记录器
        nshow: 若为整数，则每隔nshow轮日志输出当前最优目标函数值
    
    Returns
    -------
    更新优化过程之后的func_opter_parms
    
    参考：
    https://github.com/7ossam81/EvoloPy
    '''
    
    # 参数提取
    opter_name = func_opter_parms.parms_opter['opter_name']
    if opter_name == '' or isnull(opter_name):
        opter_name  = 'HHO'
    func_opter_parms.parms_opter['opter_name'] = opter_name
    # 目标函数参数
    x_lb = func_opter_parms.parms_func['x_lb']
    x_ub = func_opter_parms.parms_func['x_ub']
    dim = func_opter_parms.parms_func['dim']
    kwargs = func_opter_parms.parms_func['kwargs']
    # 优化器参数
    PopSize = func_opter_parms.parms_opter['PopSize']
    Niter = func_opter_parms.parms_opter['Niter']
    beta = func_opter_parms.parms_opter['beta']
    alpha = func_opter_parms.parms_opter['alpha']
    # 日志参数
    logger = func_opter_parms.parms_log['logger']
    nshow = func_opter_parms.parms_log['nshow']
    
    
    # 边界统一为列表
    if not isinstance(x_lb, list):
        x_lb = [x_lb] * dim
    if not isinstance(x_ub, list):
        x_ub = [x_ub] * dim
    x_lb_, x_ub_ = np.array(x_lb), np.array(x_ub)
        
    
    # 初始化
    X = rand_init(PopSize, dim, x_lb, x_ub)
    
    gBest = np.zeros(dim) # 全局最优解
    gBestVal = np.inf # 全局最优值
    
    # 保存收敛过程
    convergence_curve = np.zeros(Niter) # 全局最优值
    convergence_curve_mean = np.zeros(Niter) # 平均值
    
    # 时间记录
    strt_tm = time.time()
    func_opter_parms.set_startTime(time.strftime('%Y-%m-%d %H:%M:%S'))
    
    
    # 迭代寻优
    for t in range(0, Niter):
        E1 = 2 * (1 - (t / Niter)) # 衰减因子
        
        fvals_mean = 0
        for i in range(0, PopSize):
            X[i, :] = np.clip(X[i, :], x_lb, x_ub) # 越界处理
            fval = objf(X[i, :], **kwargs) # 目标函数值
            fvals_mean = (fvals_mean*i + fval) / (i+1)

            # 更新兔子位置（最优解）
            if fval < gBestVal:
                gBestVal = fval
                gBest = X[i, :].copy()
                
            # 种群更新
            E0 = 2 * random.random() - 1 # -1 < E0 < 1
            EE = E1 *  E0 # HHO.pdf Eq.(3)
            
            # 探索策略（Exploration phase），HHO.pdf Eq.(1)
            if abs(EE) >= 1:
                q = random.random()                
                if q >= 0.5:
                    # 随机选择在别的样本基础上进行探索
                    rnd_idx = math.floor(PopSize * random.random())
                    X_rnd = X[rnd_idx, :]
                    r1 = random.random()
                    r2 = random.random()
                    X[i, :] = X_rnd - r1 * abs(X_rnd - 2 * r2 * X[i, :])
                elif q < 0.5:
                    # 在最优样本上和整体样本均值基础上进行探索
                    r3 = random.random()
                    r4 = random.random()
                    Xm = X.mean(axis=0)
                    X[i, :] = gBest - Xm - r3 * (x_lb_ + r4 * (x_ub_ - x_lb_))

            # 包围策略（Exploitation phase）
            elif abs(EE) < 1:
                r = random.random()

                if r >= 0.5 and abs(EE) < 0.5:
                    # HHO.pdf Eq.(6)
                    X[i, :] = gBest - EE * abs(gBest - X[i, :])

                elif r >= 0.5 and abs(EE) >= 0.5:
                    # HHO.pdf Eq.(4)
                    J = 2 * (1 - random.random()) # 随机跳跃幅度
                    X[i, :] = gBest - X[i, :] - EE * abs(J * gBest - X[i, :])
                    
                elif r < 0.5 and abs(EE) >= 0.5: # HHO.pdf Eq.(10)                    
                    J = 2 * (1 - random.random()) # HHO.pdf Eq.(7)
                    Y = gBest - EE * abs(J * gBest - X[i, :])
                    Y = np.clip(Y, x_lb, x_ub)
                    if objf(Y, **kwargs) < fval:
                        X[i, :] = Y.copy()
                    else:
                        # HHO.pdf Eq.(8)
                        S = np.random.randn(dim)
                        Z = Y + S * Levy(dim, beta=beta, alpha=alpha)
                        Z = np.clip(Z, x_lb, x_ub)
                        if objf(Z, **kwargs) < fval:
                            X[i, :] = Z.copy()
                            
                elif r < 0.5 and abs(EE) < 0.5: # HHO.pdf Eq.(11)
                    J = 2 * (1 - random.random())
                    # HHO.pdf Eq.(12)
                    Y = gBest - EE * abs(J * gBest - X.mean(0))
                    Y = np.clip(Y, x_lb, x_ub)
                    if objf(Y, **kwargs) < fval:
                        X[i, :] = Y.copy()
                    else:
                        # HHO.pdf Eq.(13) 
                        S = np.random.randn(dim)
                        Z = Y + S * Levy(dim, beta=beta, alpha=alpha)
                        Z = np.clip(Z, x_lb, x_ub)
                        if objf(Z, **kwargs) < fval:
                            X[i, :] = Z.copy()
           
        # 每轮迭代都保存最优目标值
        convergence_curve[t] = gBestVal
        convergence_curve_mean[t] = fvals_mean
    
        if nshow:            
           if (t+1) % nshow ==0:
               opter_name = func_opter_parms.parms_opter['opter_name']
               func_name = func_opter_parms.parms_func['func_name']
               logger.info(f'{opter_name} for {func_name}, iter: {t+1}, ' + \
                           f'best fval: {gBestVal}')
    
    
    # 更新func_opter_parms
    end_tm = time.time()  
    func_opter_parms.set_endTime(time.strftime('%Y-%m-%d %H:%M:%S'))
    func_opter_parms.set_exeTime(end_tm-strt_tm)
    func_opter_parms.set_convergence_curve(convergence_curve)
    func_opter_parms.set_convergence_curve_mean(convergence_curve_mean)
    func_opter_parms.set_best_val(gBestVal)
    func_opter_parms.set_best_x(gBest)

    return func_opter_parms
         
    
if __name__ == '__main__':
    strt_tm = time.time()
    
    objf = TestFuncs.F5
    parms_func = {'func_name': objf.__name__,
                  'x_lb': -30, 'x_ub': 30, 'dim': 50, 'kwargs': {}}
    parms_opter = {'opter_name': 'HHO-test',
                   'PopSize': 30, 'Niter': 1000,
                   'beta': 1.5, 'alpha': 0.01}
    # logger = simple_logger()
    logger = get_logger('./test/HHO_test.txt', screen_show=True)
    # parms_log = {'logger': logger, 'nshow': 10}
    parms_log = {'logger': logger, 'nshow': 100}
    
    func_opter_parms = FuncOpterInfo(parms_func, parms_opter, parms_log)
    func_opter_parms = HHO(objf, func_opter_parms)
    
    vals = pd.DataFrame({'fval_best': func_opter_parms.convergence_curve,
                         'fval_mean': func_opter_parms.convergence_curve_mean})
    plot_Series(vals, {'fval_best': '-r', 'fval_mean': '-b'}, figsize=(8.5, 6))
    
    # best_x = func_opter_parms.best_x
    # func_opter_parms.parms_log['logger'].info(f'best x: {best_x}')
    
    close_log_file(logger)
    
    
    print(f'used time: {round(time.time()-strt_tm, 6)}s.')
    