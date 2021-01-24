# -*- coding: utf-8 -*-

import time
import random
import numpy as np
import pandas as pd
from test_funcs import TestFuncs
from utils_Heuristic import FuncOpterInfo, rand_init
from utils_hoo.utils_plot.plot_Common import plot_Series
from utils_hoo.utils_general import simple_logger, isnull
from utils_hoo.utils_logging.logger_general import get_logger
from utils_hoo.utils_logging.logger_utils import close_log_file


def GWO(objf, func_opter_parms):
    '''
    todo: 目前仅考虑自变量连续实数情况，后面可增加自变量为离散的情况
    
    灰狼优化算法(Grey Wolf Optimizer) GWO algorithm
    
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
    parms_log: 日志参数信息dict，key须包含
        logger: 日志记录器
        nshow: 若为整数，则每隔nshow轮日志输出当前最优目标函数值
    
    Returns
    -------
    更新优化过程之后的func_opter_parms
    
    参考：
    GWO.pdf
    https://github.com/7ossam81/EvoloPy
    '''
    
    # 参数提取
    opter_name = func_opter_parms.parms_opter['opter_name']
    if opter_name == '' or isnull(opter_name):
        opter_name  = 'GWO'
    func_opter_parms.parms_opter['opter_name'] = opter_name
    # 目标函数参数
    x_lb = func_opter_parms.parms_func['x_lb']
    x_ub = func_opter_parms.parms_func['x_ub']
    dim = func_opter_parms.parms_func['dim']
    kwargs = func_opter_parms.parms_func['kwargs']
    # 优化器参数
    PopSize = func_opter_parms.parms_opter['PopSize']
    Niter = func_opter_parms.parms_opter['Niter']
    # 日志参数
    logger = func_opter_parms.parms_log['logger']
    nshow = func_opter_parms.parms_log['nshow']
    
    
    # 边界统一为列表
    if not isinstance(x_lb, list):
        x_lb = [x_lb] * dim
    if not isinstance(x_ub, list):
        x_ub = [x_ub] * dim
        
    
    # 初始化alpha, beta, delta
    AlphaPos = np.zeros(dim)
    AlphaVal = float('inf')
    
    BetaPos = np.zeros(dim)
    BetaVal = float('inf')
    
    DeltaPos = np.zeros(dim)
    DeltaVal = float('inf')
    
    # 初始化所有个体|样本（包括Omegas）
    pos = rand_init(PopSize, dim, x_lb, x_ub) # 样本（个体）随机初始化

    # 保存收敛过程
    convergence_curve = np.zeros(Niter) # 全局最优值
    convergence_curve_mean = np.zeros(Niter) # 平均值
    
    # 时间记录
    strt_tm = time.time()
    func_opter_parms.set_startTime(time.strftime('%Y-%m-%d %H:%M:%S'))
    
    
    # 迭代寻优
    for l in range(0, Niter):
        # 位置过界处理
        pos = np.clip(pos, x_lb, x_ub)
        
        fvals_mean = 0
        for i in range(0, PopSize):
            fval = objf(pos[i, :], **kwargs) # 目标函数值
            fvals_mean = (fvals_mean*i + fval) / (i+1)
            
            # 更新Alpha, Beta, Delta
            # Alpha存放最优解，Beta存放次优解，Delta存放再次优解
            if fval < AlphaVal:
                # DeltaVal = BetaVal
                # DeltaPos = BetaPos.copy()
                # BetaVal = AlphaVal
                # BetaPos = AlphaPos.copy()
                AlphaVal = fval
                AlphaPos = pos[i, :].copy()                
            if AlphaVal < fval < BetaVal:
                # DeltaVal = BetaVal
                # DeltaPos = BetaPos.copy()
                BetaVal = fval
                BetaPos = pos[i,:].copy()
            if BetaVal < fval < DeltaVal:                
                DeltaVal = fval
                DeltaPos = pos[i,:].copy()
        
        # 更新所有个体|样本
        a = 2 - l * (2 / Niter) # a从2线性衰减到0
        
        r1 = np.random.random(size=(PopSize, dim)) # r1和r2取(0, 1)随机数
        r2 = np.random.random(size=(PopSize, dim))
        A1 = 2 * a * r1 - a # GWO.pdf (3.3)
        C1 = 2 * r2 # GWO.pdf (3.4)
        D_alpha = abs(C1 * AlphaPos - pos) # GWO.pdf (3.5)
        X1 = AlphaPos - A1 * D_alpha # GWO.pdf (3.6)
        
        r1 = np.random.random(size=(PopSize, dim))
        r2 = np.random.random(size=(PopSize, dim))
        A2 = 2 * a * r1 - a
        C2 = 2 * r2
        D_beta = abs(C2 * BetaPos - pos)
        X2 = BetaPos - A2 * D_beta
        
        r1 = np.random.random(size=(PopSize, dim))
        r2 = np.random.random(size=(PopSize, dim))
        A3 = 2 * a * r1 - a
        C3 = 2 * r2        
        D_delta = abs(C3 * DeltaPos - pos)
        X3 = DeltaPos - A3 * D_delta
        
        pos = (X1 + X2 + X3) / 3 # GWO.pdf (3.7)
        
        # # 更新所有个体|样本
        # a = 2 - l * (2 / Niter) # a从2线性衰减到0
        
        # for i in range(0, PopSize):
        #     for j in range (0, dim): 
        #         r1 = random.random() # r1 is a random number in [0,1]
        #         r2 = random.random() # r2 is a random number in [0,1]                
        #         A1 = 2 * a * r1 - a # GWO.pdf (3.3)
        #         C1 = 2 * r2 # GWO.pdf (3.4)                
        #         D_alpha = abs(C1 * AlphaPos[j] - pos[i, j]) # GWO.pdf (3.5)
        #         X1 = AlphaPos[j] - A1 * D_alpha # GWO.pdf (3.6)
                           
        #         r1 = random.random()
        #         r2 = random.random()                
        #         A2 = 2 * a * r1 - a
        #         C2 = 2 * r2                
        #         D_beta = abs(C2 * BetaPos[j] - pos[i, j])
        #         X2 = BetaPos[j] - A2 * D_beta
                
        #         r1 = random.random()
        #         r2 = random.random()
        #         A3 = 2 * a * r1 - a
        #         C3 = 2 * r2
        #         D_delta = abs(C3 * DeltaPos[j] - pos[i, j])
        #         X3 = DeltaPos[j] - A3 * D_delta
                
        #         pos[i, j] = (X1 + X2 + X3) / 3
        
        
        # 每轮迭代都保存最优目标值
        convergence_curve[l] = AlphaVal
        convergence_curve_mean[l] = fvals_mean
      
        if nshow:            
            if (l+1) % nshow ==0:
                opter_name = func_opter_parms.parms_opter['opter_name']
                func_name = func_opter_parms.parms_func['func_name']
                logger.info(f'{opter_name} for {func_name}, iter: {l+1}, ' + \
                            f'best fval: {AlphaVal}')
    
    
    # 更新func_opter_parms
    end_tm = time.time()  
    func_opter_parms.set_endTime(time.strftime('%Y-%m-%d %H:%M:%S'))
    func_opter_parms.set_exeTime(end_tm-strt_tm)
    func_opter_parms.set_convergence_curve(convergence_curve)
    func_opter_parms.set_convergence_curve_mean(convergence_curve_mean)
    func_opter_parms.set_best_val(AlphaVal)
    func_opter_parms.set_best_x(AlphaPos)

    return func_opter_parms
         
    
if __name__ == '__main__':
    strt_tm = time.time()
    
    objf = TestFuncs.F5
    parms_func = {'func_name': objf.__name__,
                  'x_lb': -10, 'x_ub': 10, 'dim': 5, 'kwargs': {}}
    parms_opter = {'opter_name': 'GWO-test',
                   'PopSize': 20, 'Niter': 1000}
    # logger = simple_logger()
    logger = get_logger('./test/GWO_test.txt', screen_show=True)
    # parms_log = {'logger': logger, 'nshow': 10}
    parms_log = {'logger': logger, 'nshow': 100}
    
    func_opter_parms = FuncOpterInfo(parms_func, parms_opter, parms_log)
    func_opter_parms = GWO(objf, func_opter_parms)
    
    vals = pd.DataFrame({'fval_best': func_opter_parms.convergence_curve,
                         'fval_mean': func_opter_parms.convergence_curve_mean})
    plot_Series(vals, {'fval_best': '-r', 'fval_mean': '-b'}, figsize=(10, 6))
    
    # best_x = func_opter_parms.best_x
    # func_opter_parms.parms_log['logger'].info(f'best x: {best_x}')
    
    close_log_file(logger)
    
    
    print(f'used time: {round(time.time()-strt_tm, 6)}.')
    