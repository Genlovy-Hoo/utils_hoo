# -*- coding: utf-8 -*-

import numpy as np
from utils_hoo.utils_general import simple_logger


def rand_init(PopSize, dim, lb, ub):
    '''
    自变量随机初始化
    
    Parameters
    ----------
    PopSize: 需要初始化的种群数（样本数）
    dim: 自变量维度数，dim的值应与lb和ub的长度相等
    lb: 自变量每个维度取值下界，list
    ub: 自变量每个维度取值上界，list
    
    Returns
    -------
    pos: 随机初始化结果，形状为PopSize * dim
    '''
    
    pos = np.zeros((PopSize, dim))
    for i in range(dim):
        pos[:, i] = np.random.uniform(lb[i], ub[i], PopSize)
        
    return pos


class FuncOpterInfo(object):
    '''保存函数参数及优化过程信息'''
    
    def __init__(self, parms_func={}, parms_opter={}, parms_log={}):
        '''
        parms_func: 目标函数信息，默认应包含'func_name', `x_lb`, `x_ub`, `dim`
        parms_opter: 优化函数需要用到的参数信息，默认应包含'opter_name', `PopSize`,
                    `Niter`
        parms_log: 寻优过程中控制打印或日志记录的参数，默认应包含`logger`, `nshow`
        '''
        
        # 目标函数信息
        parms_func_default = {'func_name': '', 'x_lb': None, 'x_ub': None,
                              'dim': None, 'kwargs': {}}
        parms_loss = {x: parms_func_default[x] \
                                for x in parms_func_default.keys() if \
                                    x not in parms_func.keys()}
        parms_func.update(parms_loss)
        self.parms_func = parms_func
        
        # 优化算法参数
        parms_opter_default = {'opter_name': '', 'PopSize': 20, 'Niter': 100}
        parms_loss = {x: parms_opter_default[x] \
                                for x in parms_opter_default.keys() if \
                                    x not in parms_opter.keys()}
        parms_opter.update(parms_loss)
        self.parms_opter = parms_opter
        
        # 日志参数
        parms_log_default = {'logger': simple_logger(), 'nshow': 10}
        parms_loss = {x: parms_log_default[x] \
                                for x in parms_log_default.keys() if \
                                    x not in parms_log.keys()}
        parms_log.update(parms_loss)
        self.parms_log = parms_log
        
        # 优化过程和结果
        self.__best_val = None # 全局最优值
        self.__best_x = [] # 全局最优解
        self.__convergence_curve = [] # 收敛曲线（每轮最优）
        self.__convergence_curve_mean = [] # 收敛曲线（每轮平均）
        self.__startTime = None # 开始时间
        self.__endTime = None # 结束时间
        self.__exeTime = None # 优化用时（单位秒）
        
    @property
    def best_val(self):
        return self.__best_val
    
    def set_best_val(self, val):
        self.__best_val = val
        
    @property
    def best_x(self):
        return self.__best_x
    
    def set_best_x(self, x):
        self.__best_x = x
        
    @property
    def convergence_curve(self):
        return self.__convergence_curve
    
    def set_convergence_curve(self, curve):
        self.__convergence_curve = curve
        
    @property
    def convergence_curve_mean(self):
        return self.__convergence_curve_mean
    
    def set_convergence_curve_mean(self, curve):
        self.__convergence_curve_mean = curve
        
    @property
    def startTime(self):
        return self.__startTime
    
    def set_startTime(self, t):
        self.__startTime = t
        
    @property
    def endTime(self):
        return self.__endTime
    
    def set_endTime(self, t):
        self.__endTime = t
        
    @property
    def exeTime(self):
        return self.__exeTime
    
    def set_exeTime(self, t):
        self.__exeTime = t
            