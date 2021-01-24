# -*- coding: utf-8 -*-

import time
import numpy as np
import pandas as pd
from test_funcs import TestFuncs
from utils_Heuristic import FuncOpterInfo, rand_init
from utils_hoo.utils_plot.plot_Common import plot_Series
from utils_hoo.utils_general import simple_logger, isnull
from utils_hoo.utils_logging.logger_general import get_logger
from utils_hoo.utils_logging.logger_utils import close_log_file


def PSO(objf, func_opter_parms):
    '''
    todo: 目前仅考虑自变量连续实数情况，后面可增加自变量为离散的情况
    
    粒子群优化算法(Particle Swarm Optimization) PSO algorithm
    
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
        
        v_maxs: 自变量每个维度单次绝对变化量上界，list或数值，为list时长度应等于dim
        w_max: 惯性因子w最大值，w用于平衡全局搜索和局部搜索，w值越大全局寻优能力更强
        w_min: 惯性因子最小值
        w_fix: 若w_fix设置为(0, 1)之间的值，则惯性因子w固定为w_fix，不进行动态更新
            默认动态更新w时采用线性递减方法
        c1, c2: 学习因子
    parms_log: 日志参数信息dict，key须包含
        logger: 日志记录器
        nshow: 若为整数，则每隔nshow轮日志输出当前最优目标函数值
    
    Returns
    -------
    更新优化过程之后的func_opter_parms
    
    参考：
    https://www.jianshu.com/p/8c0260c21af4
    https://github.com/7ossam81/EvoloPy
    '''
    
    # 参数提取
    opter_name = func_opter_parms.parms_opter['opter_name']
    if opter_name == '' or isnull(opter_name):
        opter_name  = 'PSO'
    func_opter_parms.parms_opter['opter_name'] = opter_name
    # 目标函数参数
    x_lb = func_opter_parms.parms_func['x_lb']
    x_ub = func_opter_parms.parms_func['x_ub']
    dim = func_opter_parms.parms_func['dim']
    kwargs = func_opter_parms.parms_func['kwargs']
    # 优化器参数
    PopSize = func_opter_parms.parms_opter['PopSize']
    Niter = func_opter_parms.parms_opter['Niter']
    v_maxs = func_opter_parms.parms_opter['v_maxs']
    w_max = func_opter_parms.parms_opter['w_max']
    w_min = func_opter_parms.parms_opter['w_min']
    w_fix = func_opter_parms.parms_opter['w_fix']
    c1 = func_opter_parms.parms_opter['c1']
    c2 = func_opter_parms.parms_opter['c2']
    # 日志参数
    logger = func_opter_parms.parms_log['logger']
    nshow = func_opter_parms.parms_log['nshow']
    
    
    # 边界统一为列表
    if not isinstance(x_lb, list):
        x_lb = [x_lb] * dim
    if not isinstance(x_ub, list):
        x_ub = [x_ub] * dim
        
    if not isinstance(v_maxs, list):
        if isnull(v_maxs):
            v_maxs = [(x_ub[_]-x_lb[_]) / 10 for _ in range(dim)]
        else:
            v_maxs = [v_maxs] * dim        
    v_mins = [-x for x in v_maxs]
        
    
    # 初始化
    vel = np.zeros((PopSize, dim)) # 初始速度
    
    pBestVals = np.zeros(PopSize) # 每个个体（样本）迭代过程中的最优值
    pBestVals.fill(float('inf')) # 最小值问题初始化为正无穷大
    
    pBest = np.zeros((PopSize, dim)) # 每个个体（样本）迭代过程中的最优解
    gBest = np.zeros(dim) # 保存全局最优解
    
    gBestVal = float('inf') # 全局最优值

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
    
            # 更新每个个体的最优解（理解为局部最优解）
            if pBestVals[i] > fval:
                pBestVals[i] = fval
                pBest[i, :] = pos[i, :].copy()
                
            # 更新全局最优解
            if gBestVal > fval:
                gBestVal = fval
                gBest = pos[i, :].copy()
        
        # 更新w（w为惯性因子，值越大全局寻优能力更强）
        if not w_fix:
            # w采用线型递减方式动态更新，也可采用其它方式更新
            w = w_max - l * ((w_max - w_min) / Niter)
        else:
            if not 0 < w_fix < 1:
                raise ValueError('固定惯性因子w范围应该在(0, 1)内！')
            w = w_fix
        
        # # 速度和位置更新
        # for i in range(0, PopSize):
        #     for j in range (0, dim):
        #         r1 = random.random()
        #         r2 = random.random()
        #         # 速度更新
        #         vel[i, j] = w * vel[i, j] + \
        #                     c1 * r1 * (pBest[i, j] - pos[i,j]) + \
        #                     c2 * r2 * (gBest[j] - pos[i, j])
        #         # 速度过界处理
        #         if vel[i, j] > v_maxs[j]:
        #             vel[i, j] = v_maxs[j]                
        #         if vel[i, j] < v_mins[j]:
        #             vel[i, j] = v_mins[j] 
        #         # 位置更新                           
        #         pos[i, j] = pos[i, j] + vel[i, j]
                
        # 速度和位置更新
        r1 = np.random.random(size=(PopSize, dim))
        r2 = np.random.random(size=(PopSize, dim))
        # 速度更新
        vel = w * vel + c1 * r1 * (pBest - pos) + c2 * r2 * (gBest - pos)
        vel = np.clip(vel, v_mins, v_maxs) # 速度过界处理
        pos = pos + vel # 位置更新
        
        # 每轮迭代都保存最优目标值
        convergence_curve[l] = gBestVal
        convergence_curve_mean[l] = fvals_mean
      
        if nshow:            
            if (l+1) % nshow ==0:
                opter_name = func_opter_parms.parms_opter['opter_name']
                func_name = func_opter_parms.parms_func['func_name']
                logger.info(f'{opter_name} for {func_name}, iter: {l+1}, ' + \
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
                  'x_lb': -10, 'x_ub': 10, 'dim': 5, 'kwargs': {}}
    parms_opter = {'opter_name': 'PSO-test',
                   'PopSize': 20, 'Niter': 1000,
                   'v_maxs': 5, 'w_max': 0.9, 'w_min': 0.2, 'w_fix': False,
                   'c1': 2, 'c2': 2}
    # logger = simple_logger()
    logger = get_logger('./test/PSO_test.txt', screen_show=True)
    # parms_log = {'logger': logger, 'nshow': 10}
    parms_log = {'logger': logger, 'nshow': 100}
    
    func_opter_parms = FuncOpterInfo(parms_func, parms_opter, parms_log)
    func_opter_parms = PSO(objf, func_opter_parms)
    
    vals = pd.DataFrame({'fval_best': func_opter_parms.convergence_curve,
                         'fval_mean': func_opter_parms.convergence_curve_mean})
    plot_Series(vals, {'fval_best': '-r', 'fval_mean': '-b'}, figsize=(10, 6))
    
    # best_x = func_opter_parms.best_x
    # func_opter_parms.parms_log['logger'].info(f'best x: {best_x}')
    
    close_log_file(logger)
    
    
    print(f'used time: {round(time.time()-strt_tm, 6)}.')
    