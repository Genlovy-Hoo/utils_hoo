# -*- coding: utf-8 -*-

import math
import numpy as np
from functools import reduce
from numpy.random import uniform


def prod(it):
    '''连乘'''
    return reduce(lambda x, y: x * y, it)


class TestFuncs(object):
    '''测试函数集，输入x须为np.array'''
    
    @staticmethod
    def F1(x):
        '''平方和'''
        return np.sum(x**2)

    @staticmethod
    def F2(x):
        '''绝对值之和加上连乘'''
        return sum(abs(x)) + prod(abs(x))
    
    @staticmethod
    def F3(x):
        '''x[0]^2 + (x[0]+x[1])^2 + (x[0]+x[1]+x[2])^2 + ...'''
        return sum([sum(x[:k]) ** 2 for k in range(len(x)+1)])
    
    @staticmethod
    def F4(x):
        '''最小绝对值'''
        return min(abs(x))
    
    @staticmethod
    def F5(x):
        '''最大绝对值'''
        return max(abs(x))
    
    @staticmethod
    def F6(x):
        d = len(x)
        part1 = 100 * (x[1:d] - (x[0:d-1] ** 2)) ** 2
        part2 = (x[0:d-1] - 1) ** 2
        return np.sum(part1 + part2)
    
    @staticmethod
    def F7(x):
        return np.sum(abs((x + 0.5)) ** 2)
    
    @staticmethod
    def F8(x):
       return sum([(k+1) * (x[k] ** 4) for k in range(len(x))]) + uniform(0, 1)
    
    @staticmethod
    def F9(x):
        return sum(-x*(np.sin(np.sqrt(abs(x)))))
    
    @staticmethod
    def F10(x):
        return np.sum(x ** 2 - 10 * np.cos(2 * math.pi * x)) + 10 * len(x)
    
    @staticmethod
    def F11(x):
        d = len(x)
        part1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / d))
        part2 = np.exp(np.sum(np.cos(2 * math.pi * x)) / d) + 20
        o = part1 - part2 + np.exp(1)
        return o
