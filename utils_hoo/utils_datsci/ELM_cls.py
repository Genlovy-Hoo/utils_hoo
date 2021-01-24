# -*- coding: utf-8 -*-

import time
import numpy as np
import pandas as pd
from inspect import isfunction
from utils_hoo.utils_general import isnull
from sklearn.preprocessing import OneHotEncoder
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split as tts
from utils_hoo.utils_datsci.utils_stats import scale_skl
from sklearn import metrics

class ELM_cls(object):
    '''
    极限学习机，分类任务
    
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
        
        # 将标签进行onehot编码
        self.Yonehot = None
        if len(Ytrain.shape) == 1 or Ytrain.shape[1] == 1:
            self.Yonehot = OneHotEncoder()
            Ytrain = self.Yonehot.fit_transform(Ytrain.reshape(-1, 1)).toarray()
        
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
        Ypre_ = np.dot(Hide, self.beta)
        
        Ypre = np.zeros_like(Ypre_)
        np.put_along_axis(Ypre, Ypre_.argmax(1), 1, 1)
        Ypre = np.asarray(Ypre, dtype=np.int8)
        
        # 返回形式与训练数据中Y统一
        if not isnull(self.Yonehot):
            Ypre = self.Yonehot.inverse_transform(Ypre).reshape(-1,)
            
        return Ypre
    
    def predict_proba(self, X):
        '''概率预测'''
        raise NotImplementedError
    
#%%
if __name__ == '__main__':
    strt_tm = time.time()
    
    #%%
    data = datasets.load_iris()
    X = pd.DataFrame(data['data'], columns=data.feature_names)
    Y = pd.Series(data['target'])
    # Y = np.array(pd.get_dummies(Y), dtype=np.int8)
   
    Xtrain, Xtest, Ytrain, Ytest = tts(X, Y, test_size=0.4,
                                       random_state=5262)
    Xtrain, [Xtest], _ = scale_skl(Xtrain, [Xtest])
    
    funcAct = 'softplus'
    w_low, w_up, b_low, b_up, = -1, 1, -1, 1
    C = None
    random_state = 5262
    
    for Nhide in range(1, 50, 2):
        mdlEML = ELM_cls(Nhide=Nhide, funcAct=funcAct,
                         w_low=w_low, w_up=w_up, b_low=b_low, b_up=b_up,
                         C=C, random_state=random_state)
        
        mdlEML = mdlEML.fit(Xtrain, Ytrain)
        
        Ypre = mdlEML.predict(Xtest)
        
        acc = metrics.accuracy_score(Ytest, Ypre)
        print('Nhide - %d, acc：%f' % (Nhide, acc))
        
        
    #%%
    print(f'used time: {round(time.time()-strt_tm, 6)}s.')