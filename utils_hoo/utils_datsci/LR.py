# -*- coding: utf-8 -*-

import time
import numpy as np
import pandas as pd
from utils_hoo.utils_general import isnull
from sklearn.linear_model import LogisticRegression as lrc

import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['font.serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt


class LR_cls_bin(object):
    '''
    逻辑回归二分类
    
    记输入为X，输出为y，则逻辑回归二分类模型表达式为：
        y` = Sigmoid(X * W  + b)，其中y`表示模型预测值，
        将截距项b并入W（在X中添加常数项变量），简写为：
        y` = Sigmoid(X * W)
        Sigmoid函数为: Sigmoid(x) = 1.0 / (1 + np.exp(-x))
    代价函数为：
        Cost = y * ln(y`) + (1-y) * ln(1-y`)
        （分类正确代价为0，分类错误代价无穷大，由极大似然估计得来）
    转为最小化问题后总代价函数为：
        Cost = -Sum(y * ln(y`) + (1-y) * ln(1-y`))
        
    梯度下降法：
    根据上面三个方程，用链式求导法则可得到Cost对W的导数：
        J = X * (y` - y)
    J即为梯度下降法中的梯度
    
    牛顿法：
        
        
    参考：
    https://www.cnblogs.com/loongofqiao/p/8642045.html
    '''

    def __init__(self, opt_method='gd', max_iter=1000, lr=0.01):
        '''
        Parameters
        ----------
        opt_method: 优化算法，默认梯度下降gd
        max_iter: 最大迭代次数
        lr: 学习速率
        '''
        
        self.w = '未初始化参数(shape: NcolX*1)'
        self.b = '未初始化参数（截距项）'
        
        self.opt_method = opt_method
        
        self.max_iter = max_iter
        self.lr = lr
        
    def forward(self, X, w, b):
        '''前向传播（模型表达式）'''
        return self.sigmoid(np.dot(X, w) + b)
        
    @staticmethod
    def sigmoid(x):
        '''sigmoid激活函数'''
        return 1.0 / (1 + np.exp(-x))
    
    @staticmethod
    def XaddConst(X):
        '''X添加常数列'''
        const = np.ones((X.shape[0], 1))
        return np.concatenate((X, const), axis=1)

    def fit(self, X_train, y_train):
        '''
        模型训练
        
        Parameters
        ----------
        X_train: 训练集输入，pd.DataFrame或np.array，每行一个样本
        y_train: 训练集输出，pd.Series或pd.DataFrame或np.array，每行一个样本
        '''
        
        X_train, y_train = np.array(X_train), np.array(y_train)
        NcolX = X_train.shape[1] # 特征数
        Xconst = self.XaddConst(X_train) # X添加常数项
        # y转化为二维
        if len(y_train.shape) == 1 or y_train.shape[1] == 1:
            y_train = y_train.reshape(-1, 1)
            
        # w和b初始化
        self.w = np.zeros((NcolX, 1))
        # 输入层——>隐藏层偏置b随机化
        self.b = 1
        
        # 梯度下降法
        if self.opt_method == 'gd':
            # 系数转为二维
            W = np.array(list(self.w[:,0]) + [self.b]).reshape(-1, 1)
            
            # 梯度下降更新W
            k = 1
            while k < self.max_iter:
                h = self.sigmoid(np.dot(Xconst, W)) # 前向传播
                grad = np.dot(Xconst.T, h - y_train) # 梯度
                W = W - self.lr * grad
                
                k += 1
                
            self.w = W[:-1, :]
            self.b = W[-1][-1]
            
        # 牛顿法
        elif self.opt_method.lower() in ['newton', 'nt']:
            # 系数转为二维
            W = np.array(list(self.w[:,0]) + [self.b]).reshape(-1, 1)
            
            # 牛顿法更新W
            k = 1
            while k < self.max_iter:
                p = self.sigmoid(np.dot(Xconst, W)) # 前向传播
                grad = np.dot(Xconst.T, p - y_train) # 梯度                
                # Hesse矩阵
                H = np.dot(Xconst.T, np.diag(p.reshape(-1,)))
                H = np.dot(H, np.diag(1 - p.reshape(-1,)))
                H = np.dot(H, Xconst)
                W = W - np.dot(np.linalg.inv(H), grad) # 梯度更新
    
                k += 1
                
            self.w = W[:-1, :]
            self.b = W[-1][-1]
        
        return self
    
    def predict_proba(self, X):
        '''概率预测'''
        y_pre_p = self.forward(X, self.w, self.b)
        return y_pre_p.reshape(-1,)

    def predict(self, X, p_cut=0.5):
        '''标签预测'''        
        y_pre_p = self.predict_proba(X)
        y_pre = (y_pre_p >= p_cut).astype(int) 
        return y_pre
    
#%%
if __name__ == '__main__':
    strt_tm = time.time()
    
    #%%
    data = pd.read_excel('./test/test_data1.xlsx')
    # data = pd.read_excel('./test/test_data2.xlsx')
    X = data[['x1', 'x2']]
    y = data['y']
    
    opt_method = 'gd'
    # opt_method = 'newton'
    
    max_iter = 10000
    lr = 0.001
    
    mdl = LR_cls_bin(opt_method=opt_method, max_iter=max_iter, lr=lr)
    
    X_train, y_train = X, y
    mdl = mdl.fit(X, y)
    print(f'梯度下降法参数结果：\nw: \n{mdl.w}\nb: {mdl.b}')
    
    def plot_result(data, w, b, title=None):
        plt.figure(figsize=(10, 7))
        data0 = data[data['y'] == 0]
        data1 = data[data['y'] == 1]
        plt.plot(data0['x1'], data0['x2'], 'ob', label='y=0')
        plt.plot(data1['x1'], data1['x2'], 'or', label='y=1')
        
        x = np.arange(data['x1'].min(), data['x1'].max(), 0.1)
        y = (-b - w[0]*x) / w[1]
        plt.plot(x, y, '-')
        
        plt.legend(loc=0)
        
        if title:
            plt.title(title)
        
        plt.show()
        
    plot_result(data, mdl.w, mdl.b, '梯度下降法')
        
    #%%
    print(f'\nused time: {round(time.time()-strt_tm, 6)}s.')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    