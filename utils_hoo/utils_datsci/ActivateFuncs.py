# -*- coding: utf-8 -*-

import numpy as np


def sigmoid(x):
    '''sigmoid激活函数'''
    return 1.0 / (1 + np.exp(-x))


def softplus(x):
    '''softplus激活函数 '''
    return np.log(1 + np.exp(x))


def tanh(x):
    '''tanh激活函数'''
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
