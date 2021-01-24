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


def calculateCost(objf, population, PopSize, x_lb, x_ub, **kwargs):        
    '''
    计算种群中每个个体的函数值
    
    Parameters
    ----------    
    objf: 目标函数，接受每个个体以及kwargs为参数
    population: 所有个体位置（所有解）
    PopSize: 种群个体数量
    x_lb, x_ub: 取值上下边界
    
    Returns
    -------
    fvals: 个体函数值列表
    '''
    
    fvals = np.full(PopSize, np.inf)
    for i in range(0,PopSize):
        # 越界处理
        population[i] = np.clip(population[i], x_lb, x_ub)
        # 个体值计算
        fvals[i] = objf(population[i, :], **kwargs) 
        
    return fvals
        

def sortPopulation(population, fvals):
    '''
    个体排序：较优值排在前面
    
    Parameters
    ---------- 
    population: 所有个体位置（所有解）
    fvals: 所有个体值列表
          
    Returns
    -------
    population: 排序后的种群
    fvals: 排序后的个体值列表
    '''
    
    sortedIndices = fvals.argsort()
    population = population[sortedIndices]
    fvals = fvals[sortedIndices]
    
    return population, fvals


def crossoverPopulaton(population, fvals, PopSize, Pcrs, Ntop, dim):
    '''
    群体交叉
    
    Parameters
    ---------- 
    population: 所有个体位置（所有解）
    fvals: 所有个体值列表
    PopSize: 种群个体数量
    Pcrs: 交叉概率
    Ntop: 最优个体保留数（不进行交叉变异的最优个体数）         
	 
    Returns
    -------
    newPopulation: 新种群位置（新解）
    '''
    
    # 新种群初始化
    newPopulation = population.copy()
    
    for i in range(Ntop, PopSize-1, 2):
        # 轮盘赌法选择待交叉个体
        parent1, parent2 = pairSelection(population, fvals, PopSize)
        parentsCrossoverProbability = random.uniform(0.0, 1.0)
        if parentsCrossoverProbability < Pcrs:
            offspring1, offspring2 = crossover(dim, parent1, parent2)
            # 更新交叉后的个体
            newPopulation[i] = offspring1
            newPopulation[i+1] = offspring2   
     
    return newPopulation
        
    
def mutatePopulaton(population, PopSize, Pmut, Ntop, x_lb, x_ub):
    '''
    群体变异
    
    Parameters
    ---------- 
    population: 所有个体位置（所有解）
    PopSize: 种群个体数量
    Pmut: 个体变异概率
    Ntop: 最优个体保留数（不进行交叉变异的最优个体数）
    x_lb, x_ub: 取值上下边界
         
    Returns
    -------
    
    '''
    
    newPopulation = population.copy()
    for i in range(Ntop, PopSize):
        # 变异操作
        offspringMutationProbability = random.uniform(0.0, 1.0)
        if offspringMutationProbability < Pmut:
            offspring = mutation(population[i], len(population[i]), x_lb, x_ub)
            newPopulation[i] = offspring
    return newPopulation


def pairSelection(population, fvals, PopSize):    
    '''
    轮盘赌法选择交叉个体对
    
    Parameters
    ----------
    population: 所有个体位置（所有解）
    fvals: 所有个体值列表
    PopSize: 种群个体数量
          
    Returns
    -------
    被选中的两个个体
    '''
    
    parent1Id = rouletteWheelSelectionId(fvals, PopSize)
    parent1 = population[parent1Id].copy()
    
    parent2Id = rouletteWheelSelectionId(fvals, PopSize)    
    parent2 = population[parent2Id].copy()
   
    return parent1, parent2

    
def rouletteWheelSelectionId(fvals, PopSize): 
    '''
    轮盘赌法：个体函数值越小（最小值问题），越容易被选中
    
    Parameters
    ---------- 
    fvals: 所有个体值列表
    PopSize: 种群个体数量
          
    Returns
    -------
    被选中的个体序号
    '''
    
    # 最小值问题转化
    reverse = max(fvals) + min(fvals)
    reverseScores = reverse - fvals
    
    sumScores = sum(reverseScores)
    pick = random.uniform(0, sumScores)
    current = 0
    for individualId in range(PopSize):
        current += reverseScores[individualId]
        if current > pick:
            return individualId

def crossover(individualLength, parent1, parent2):
    '''
    两个个体交叉操作
    
    Parameters
    ---------- 
    individualLength: 个体长度（维度）
    parent1, parent2: 待交叉个体
          
    Returns
    -------
    交叉操作后的两个新个体
    '''
    
    # 选择交叉位置
    crossover_point = random.randint(0, individualLength-1)
    # 以交叉位置为切分，新个体的前半部分取个体1，后半部分取个体2
    offspring1 = np.concatenate([parent1[0:crossover_point],
                                 parent2[crossover_point:]])    
    offspring2 = np.concatenate([parent2[0:crossover_point],
                                 parent1[crossover_point:]])
    
    return offspring1, offspring2


def mutation(offspring, individualLength, x_lb, x_ub):
    '''
    个体变异操作
    
    Parameters
    ---------- 
    offspring: 待变异个体
    individualLength: 个体长度
    x_lb, x_ub: 取值上下边界
         
    Returns
    -------
    返回变异后的个体
    '''
    
    # 随机选择变异位置，随机取变异值
    mutationIndex = random.randint(0, individualLength-1)
    mutationValue = random.uniform(x_lb[mutationIndex], x_ub[mutationIndex])
    offspring[mutationIndex] = mutationValue
    return offspring


def clearDups(population, dim, x_lb, x_ub):  
    '''
    替换重复个体
    
    Parameters
    ----------    
    population: 所有个体位置（所有解）
    x_lb, x_ub: 取值上下边界
    
    Returns
    -------
    随机替换重复值后的新种群
    '''
    
    newPopulation = np.unique(population, axis=0)
    oldLen = len(population)
    newLen = len(newPopulation)
    if newLen < oldLen:
        nDuplicates = oldLen - newLen
        newIndividuals = rand_init(nDuplicates, dim, x_lb, x_ub)
        newPopulation = np.append(newPopulation, newIndividuals, axis=0)
            		        
    return newPopulation


def GA(objf, func_opter_parms):
    '''
    todo: 目前仅考虑自变量连续实数情况，后面可增加自变量为离散的情况
    
    遗传算法(Genetic Algorithm) GA（实数编码）
    
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
        
        Pcrs: 交叉概率
        Pmut: 变异概率
        Ntop: 每一轮（代）保留的最优个体数
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
        opter_name  = 'GA'
    func_opter_parms.parms_opter['opter_name'] = opter_name
    # 目标函数参数
    x_lb = func_opter_parms.parms_func['x_lb']
    x_ub = func_opter_parms.parms_func['x_ub']
    dim = func_opter_parms.parms_func['dim']
    kwargs = func_opter_parms.parms_func['kwargs']
    # 优化器参数
    PopSize = func_opter_parms.parms_opter['PopSize']
    Niter = func_opter_parms.parms_opter['Niter']
    Pcrs = func_opter_parms.parms_opter['Pcrs']
    Pmut = func_opter_parms.parms_opter['Pmut']
    Ntop = func_opter_parms.parms_opter['Ntop']
    # 日志参数
    logger = func_opter_parms.parms_log['logger']
    nshow = func_opter_parms.parms_log['nshow']
    
    
    # 边界统一为列表
    if not isinstance(x_lb, list):
        x_lb = [x_lb] * dim
    if not isinstance(x_ub, list):
        x_ub = [x_ub] * dim
    
    # 全局最优解和全局最优值
    gBest = np.zeros(dim)
    gBestVal = float('inf')
    
    population = rand_init(PopSize, dim, x_lb, x_ub) # 样本（个体）随机初始化
    fvals = np.random.uniform(0.0, 1.0, PopSize) # 个体函数值
    
    convergence_curve = np.zeros(Niter) # 全局最优值
    convergence_curve_mean = np.zeros(Niter) # 平均值
    
    # 时间记录
    strt_tm = time.time()
    func_opter_parms.set_startTime(time.strftime('%Y-%m-%d %H:%M:%S'))
    
    for l in range(Niter):
        
        # 计算个体值
        fvals = calculateCost(objf, population, PopSize, x_lb, x_ub, **kwargs)            
        
        # 个体排序
        population, fvals = sortPopulation(population, fvals)
        
        # 最优解纪录
        gBestVal = fvals[0]
        gBest = population[0]

        # 交叉
        population = crossoverPopulaton(population, fvals, PopSize, Pcrs, Ntop,
                                        dim)           
        # 变异
        population = mutatePopulaton(population, PopSize, Pmut, Ntop,
                                     x_lb, x_ub)
        # 重复值处理
        population = clearDups(population, dim, x_lb, x_ub)
        
        # 每轮迭代都保存最优目标值
        convergence_curve[l] = gBestVal
        convergence_curve_mean[l] = np.mean(fvals)
      
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
    
    objf = TestFuncs.F1
    parms_func = {'func_name': objf.__name__,
                  'x_lb': -10, 'x_ub': 10, 'dim': 5, 'kwargs': {}}
    parms_opter = {'opter_name': 'GA-test',
                   'PopSize': 20, 'Niter': 1000,
                   'Pcrs': 0.7, 'Pmut': 0.1, 'Ntop': 2}
    # logger = simple_logger()
    logger = get_logger('./test/GA_test.txt', screen_show=True)
    # parms_log = {'logger': logger, 'nshow': 10}
    parms_log = {'logger': logger, 'nshow': 100}
    
    func_opter_parms = FuncOpterInfo(parms_func, parms_opter, parms_log)
    func_opter_parms = GA(objf, func_opter_parms)
    
    vals = pd.DataFrame({'fval_best': func_opter_parms.convergence_curve,
                         'fval_mean': func_opter_parms.convergence_curve_mean})
    plot_Series(vals, {'fval_best': '-r', 'fval_mean': '-b'}, figsize=(10, 6),
                title='GA优化目标函数值收敛过程')
    
    # best_x = func_opter_parms.best_x
    # func_opter_parms.parms_log['logger'].info(f'best x: {best_x}')
    
    close_log_file(logger)
    
    
    print(f'used time: {round(time.time()-strt_tm, 6)}.')
    