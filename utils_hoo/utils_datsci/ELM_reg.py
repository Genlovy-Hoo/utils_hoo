# -*- coding: utf-8 -*-

import time
import numpy as np
import pandas as pd
from inspect import isfunction
from utils_hoo.utils_general import isnull
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split as tts
from utils_hoo.utils_datsci.utils_stats import mape

import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['font.serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

class ELM_reg(object):
    '''
    极限学习机，回归任务
    
    记输入为层X，输出层为Y，隐藏层为H，样本量为Nsmp、X特征数为NcolX、
    隐藏层节点数为Nhide、Y特征数为NcolY，则ELM的过程为：
      H(Nsmp*Nhide) = X(Nsmp*NcolX) * w(NcolX*Nhide) + b((Nsmp*1)*Nhide)
      Y(Nsmp*NcolY) = H(Nsmp*Nhide) * beta(Nhide*NcolY)
    ELM的训练过程：W和b随机生成，beta则利用公式求解析解(beta = H的MP广义逆 * Y)
    
    参考：
    https://blog.csdn.net/m0_37922734/article/details/80424133
    https://blog.csdn.net/qq_32892383/article/details/90760481
    https://blog.csdn.net/qq_40360172/article/details/105175946
    '''

    def __init__(self, Nhide=10, funcAct='softplus', 
                 w_low=-1, w_up=1, b_low=-1, b_up=1,
                 C=None, random_state=5262):
        '''
        Parameters
        ----------
        Nhide: 隐层节点数
        funcAct: 激活函数，可选['softplus', 'sigmoid', 'tanh']或自定义
        w_low, w_up: 输入层—>隐层权重w取值下限和上限
        b_low, b_up: 输入层—>隐层偏置项b取值下限和上限
        C: 正则化参数？
        random_state: 随机数种子
        '''
        
        self.Nhide = Nhide # 隐藏层节点数
        
        # 输入层—>隐层权重w和偏置项b取值上下限
        self.w_low = w_low
        self.w_up = w_up
        self.b_low = b_low
        self.b_up = b_up
        
        self.w = f'未初始化参数(shape: NcolX*{Nhide})'
        self.b = f'未初始化参数(shape: 1*{Nhide})'
        self.beta = f'未初始化参数(shape: {Nhide}*NcolY)'
        
        # 正则化参数
        self.C = C
        
        # 激活函数
        if funcAct == 'softplus':
            self.funcAct = self.softplus
        elif funcAct == 'sigmoid':
            self.funcAct = self.sigmoid
        elif funcAct == 'tanh':
            self.funcAct = self.tanh
        else:
            if isfunction(funcAct):
                self.funcAct = funcAct
            else:
                raise ValueError('不能识别的激活函数，请检查！')
        
        # 其他参数
        self.random_state = random_state
        
    @staticmethod
    def sigmoid(x):
        '''sigmoid激活函数'''
        return 1.0 / (1 + np.exp(-x))

    @staticmethod
    def softplus(x):
        '''softplus激活函数 '''
        return np.log(1 + np.exp(x))

    @staticmethod
    def tanh(x):
        '''tanh激活函数'''
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))  

    def fit(self, Xtrain, Ytrain):
        '''
        模型训练
        
        Parameters
        ----------
        Xtrain: 训练集输入，pd.DataFrame或np.array，每行一个样本
        Ytrain: 训练集输出，pd.DataFrame或np.array，每行一个样本
        '''
        
        Xtrain, Ytrain = np.array(Xtrain), np.array(Ytrain)
        Nsmp, NcolX = Xtrain.shape[0], Xtrain.shape[1] # 样本数和特征数
        
        # 将标签转化为二维
        self.Yreshape = False
        if len(Ytrain.shape) == 1:
            self.Yreshape = True
            Ytrain = Ytrain.reshape(-1, 1)
        
        # 随机数种子
        if isnull(self.random_state):
            rnd_w = np.random.RandomState()
            rnd_b = np.random.RandomState()
        else:
            rnd_w = np.random.RandomState(self.random_state)
            rnd_b = np.random.RandomState(self.random_state)
            
        # 输入层——>隐藏层权重w随机化
        self.w = rnd_w.uniform(self.w_low, self.w_up, (NcolX, self.Nhide))
        # 输入层——>隐藏层偏置b随机化
        self.b = rnd_b.uniform(self.b_low, self.b_up, (1, self.Nhide))
        Bhide= np.ones([Nsmp, self.Nhide]) * self.b
        
        # 隐层计算
        Hide = np.matrix(self.funcAct(np.dot(Xtrain, self.w) + Bhide))
        
        # beta计算
        if isnull(self.C):
            iMP = np.linalg.pinv(Hide) # Moore–Penrose广义逆
            self.beta = np.dot(iMP, Ytrain)
        else:
            Hide_ = np.dot(Hide.T, Hide) + Nsmp / self.C
            iMP = np.linalg.pinv(Hide_) # Moore–Penrose广义逆
            iMP_ = np.dot(iMP, Hide.T)
            self.beta = np.dot(iMP_, Ytrain)
        
        return self

    def predict(self, X):
        '''预测'''
        
        Nsmp = X.shape[0]
        Bhide = np.ones([Nsmp, self.Nhide]) * self.b
        Hide = np.matrix(self.funcAct(np.dot(X, self.w) + Bhide))
        Ypre = np.array(np.dot(Hide, self.beta))
        
        if self.Yreshape:
            Ypre = Ypre.reshape(-1)
            
        return Ypre
    
#%%
if __name__ == '__main__':
    strt_tm = time.time()
    
    #%%
    # 产生数据集
    X = np.linspace(0, 20, 200)
    noise = np.random.normal(0, 0.08, 200)
    y = np.sin(X) + np.cos(0.5*X) + noise
    # 转化成二维形式
    X = np.array(X).reshape(-1, 1)  
    
    funcAct = 'softplus'
    w_low, w_up, b_low, b_up, = -1, 1, -1, 1
    C = None
    random_state = 5262
    
    Xtest = np.linspace(0, 20, 200).reshape(-1, 1)
    Ytest = np.sin(Xtest) + np.cos(0.5*Xtest)
    Ytest = Ytest.reshape(-1)
    plt.figure(figsize=(12, 7))   
    plt.plot(Xtest, Ytest, 'or', label='ori') # 原始数据散点图 
    # plt.plot(X, y, 'or', label='ori') # 原始数据散点图 
    color = ['g', 'b', 'y', 'c', 'm'] # 不同隐藏层线条设置不同的颜色    
    # 比较不同隐藏层拟合效果
    j=0
    for i in range(5, 30, 5):
        mdlELM = ELM_reg(Nhide=i, funcAct=funcAct,
                         w_low=w_low, w_up=w_up, b_low=b_low, b_up=b_up,
                         C=C, random_state=random_state)
        mdlELM = mdlELM.fit(X, y)        
        Ypre = mdlELM.predict(Xtest)
        plt.plot(Xtest, Ypre, color[j], label='Nhide_'+str(i))
        plt.title('ELM regression test')
        plt.xlabel('x')
        plt.ylabel('y')
        j+=1        
        
        vMAPE = mape(Ytest, Ypre)
        print('Nhide - %d, mape：%f' % (i, vMAPE))
    plt.legend(loc=0)
    plt.show()
    
    #%%
    # data = datasets.load_boston()
    data = datasets.load_diabetes()
    X = pd.DataFrame(data['data'], columns=data.feature_names)
    # Y = pd.DataFrame(data['target'], columns=['y'])
    Y = pd.Series(data['target'])
   
    Xtrain, Xtest, Ytrain, Ytest = tts(X, Y, test_size=0.4,
                                        random_state=5262)
    
    funcAct = 'softplus'
    w_low, w_up, b_low, b_up, = -1, 1, -1, 1
    C = None
    random_state = 5262
    
    for Nhide in range(1, 50, 2):
        mdlELM = ELM_reg(Nhide=Nhide, funcAct=funcAct,
                          w_low=w_low, w_up=w_up, b_low=b_low, b_up=b_up,
                          C=C, random_state=random_state)
        
        mdlELM = mdlELM.fit(Xtrain, Ytrain)
        
        Ypre = mdlELM.predict(Xtest)
        
        vMAPE = mape(Ytest, Ypre)
        print('Nhide - %d, mape：%f' % (Nhide, vMAPE))
        
    #     # plt.figure(figsize=(12, 7))
    #     # plt.plot(Ytest.reset_index(drop=True), '.-b', label='Ytrue')
    #     # plt.plot(Ypre, '.-r', label='Ypre')
    #     # plt.legend(loc=0)
    #     # plt.show()
        
    #%%
    print(f'used time: {round(time.time()-strt_tm, 6)}s.')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    