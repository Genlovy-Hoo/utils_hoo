# -*- coding: utf-8 -*-

import time
import pandas as pd
import statsmodels.api as sm
from utils_hoo.utils_datsci.utils_stats import scale_skl
from sklearn.datasets import load_iris, load_diabetes
from sklearn.linear_model import LinearRegression as lr
from sklearn.linear_model import LogisticRegression as lrc
from keras import Input
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD


def linearFit(X, y):
    '''
    statsmodels线性回归
    '''
    
    X = sm.add_constant(X) # 添加截距项
    mdl = sm.OLS(y, X).fit()
    
    Params = mdl.params
    R2, R2adj = mdl.rsquared, mdl.rsquared_adj
    
    y_pre = mdl.predict(X)
    
    return Params, R2, R2adj, y_pre, mdl


def lr_skl(X, y):
    '''
    sklearn线性回归
    '''
    
    mdl = lr().fit(X, y)
    
    return mdl


def lr_keras(X, y):
    '''
    keras线性回归
    '''
    
    mdl = Sequential()
    mdl.add(Input(shape=(X.shape[1],)))
    mdl.add(Dense(1))
    
    mdl.compile(loss='mse', optimizer=SGD(learning_rate=0.01, momentum=0.1))
    mdl.fit(X, y, epochs=500, batch_size=16, verbose=0)
    
    return mdl


def lr_torch(X, y):
    '''pytorch线性回归'''
    raise NotImplementedError


def lr_tf(X, y):
    '''tensorflow线性回归'''
    raise NotImplementedError
    
    
def lrc_skl(X, y):
    '''
    sklearn逻辑回归
    '''
    
    mdl = lrc().fit(X, y)
    
    return mdl


if __name__ == '__main__':
    strt_tm = time.time()
    
    # 回归数据集
    data_reg = load_diabetes()
    Xreg, yreg = data_reg['data'], data_reg['target']
    data_reg = pd.DataFrame(Xreg,
                          columns=['x_'+str(_) for _ in range(Xreg.shape[1])])
    data_reg['y'] = yreg
    data_reg = scale_skl(data_reg)[0]
    Xreg = data_reg[['x_'+str(_) for _ in range(Xreg.shape[1])]]
    yreg = data_reg['y']
    
    # 线性回归模型比较
    Params, R2, R2adj, y_pre, mdl_sm = linearFit(Xreg, yreg)
    mdl_lr_skl = lr_skl(Xreg, yreg)
    mdl_lr_keras = lr_keras(Xreg, yreg)
    w_keras = mdl_lr_keras.weights[0].numpy()
    w_skl = mdl_lr_skl.coef_
    W = pd.DataFrame(Params.iloc[1:], columns=['w_sm'])
    W['w_skl'] = w_skl
    W['w_keras'] = w_keras
    W.loc['const', 'w_keras'] = mdl_lr_keras.weights[1].numpy()[0]
    W.loc['const', 'w_skl'] = mdl_lr_skl.intercept_
    W.loc['const', 'w_sm'] = Params.iloc[0]
    print(W.round(6))
    
    
    # 分类数据集
    data_cls = load_iris()
    Xcls, ycls = data_cls['data'], data_cls['target']
    Xcls = pd.DataFrame(Xcls,
                        columns=['x_'+str(_) for _ in range(Xcls.shape[1])])
    Xcls = scale_skl(Xcls)[0]
    
    # 逻辑回归模型比较
    mdl_lrc_skl = lrc_skl(Xcls, ycls)
    
    
    print(f'used time: {round(time.time()-strt_tm, 6)}s.')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    