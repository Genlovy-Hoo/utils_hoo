# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from utils_hoo.utils_general import isnull
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import norm, lognorm, weibull_min, kstest
from sklearn.linear_model import LinearRegression as lr


def normPdf(mu, sigma, x):
    '''正态分布概率密度函数'''
    p = norm(mu, sigma).pdf(x)
    # p = np.exp(-((x - mu)**2)/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
    return p


def normFit(series):
    '''对series（pd.Series或np.array）进行正态分布参数估计'''    
    # mu = series.mean()
    # sigma = series.std(ddof=0)
    mu, sigma = norm.fit(series)    
    return mu, sigma


def Fit_norm(series, x):
    '''
    用series（pd.Series或np.array）拟合一个正态分布，并计算在该分布下取值为x的概率
    '''
    mu, sigma = normFit(series)
    return normPdf(mu, sigma, x)


def normTest(series):
    '''
    对series（pd.Series或np.array）进行正态分布检验（目前采用KS检验）
    '''    
    mu, sigma = normFit(series)
    Stat, P = kstest(series, 'norm', args=(mu, sigma))    
    return (Stat, P), (mu, sigma)


def lognormPdf(mu, sigma, x):
    '''对数正态分布概率密度函数'''
    p = lognorm(s=sigma, loc=0, scale=np.exp(mu)).pdf(x)
    # p = np.exp(-((np.log(x) - mu)**2)/(2 * sigma**2)) / \
    #                                           (x * sigma * np.sqrt(2*np.pi))
    return p


def lognormFit(series):
    '''对series（pd.Series或np.array）进行对数正态分布参数估计'''
    # stats中lognorm分布参数估计和lognormPdf中参数关系：
    # 若设置floc=0（即始终loc=0），则有s = sigma，scale = e ^ mu
    
    # mu = np.log(series).mean()
    # sigma = np.log(series).std(ddof=0)
    
    s, loc, scale = lognorm.fit(series, floc=0)   
    sigma, mu = s, np.log(scale) 
    
    return mu, sigma


def Fit_lognorm(series, x):
    '''
    用series（pd.Series或np.array）拟合一个对数正态分布，并计算在该分布下取值为x的概率
    '''
    mu, sigma = lognormFit(series)
    return lognormPdf(mu, sigma, x)


def lognormTest(series):
    '''
    对series（pd.Series或np.array）进行对数正态分布检验（目前采用KS检验）
    '''
    mu, sigma = lognormFit(series)
    s, loc, scale = sigma, 0, np.exp(mu)
    Stat, P = kstest(series, 'lognorm', args=(s, loc, scale))
    return (Stat, P), (mu, sigma)


def weibullPdf(k, lmd, x):
    '''威布尔分布概率密度函数'''
    p = weibull_min(c=k, loc=0, scale=lmd).pdf(x)
    # p = (k/lmd) * (x/lmd)**(k-1) * np.exp(-(x/lmd)**k)
    return p


def weibullFit(series):
    '''对series（pd.Series或np.array）进行威布尔分布参数估计'''
    # stats中weibull_min分布参数估计和weibullPdf中weibull分布参数关系：
    # 若设置floc=0（即始终loc=0），则有c = k，scale = lmd
    # stats中weibull_min分布参数估计和np.random.weibull分布参数关系：
    # 若设置floc=0（即始终loc=0），则有c = a，scale = 1
    
    c, loc, scale = weibull_min.fit(series, floc=0)
    k, lmd = c, scale
    
    return k, lmd


def Fit_weibull(series, x):
    '''
    用series（pd.Series或np.array）拟合一个威布尔分布，并计算在该分布下取值为x的概率
    '''
    k, lmd = weibullFit(series)
    return weibullPdf(k, lmd, x)


def weibullTest(series):
    '''
    对series（pd.Series或np.array）进行威布尔分布检验（目前采用KS检验）
    '''
    k, lmd = weibullFit(series)
    c, loc, scale = k, 0, lmd
    Stat, P = kstest(np.log(series), 'weibull_min', args=(c, loc, scale))
    return (Stat, P), (k, lmd)


def scale_skl(df_fit, dfs_trans=None, cols=None,
              scale_type='std', **kwargs):
    '''
    sklearn数据标准化处理，对df_fit中指定列cols进行标准化
    scale_type指定标准化类型具体如下：
        'std'或'z-score'使用StandardScaler
        'maxmin'或'minmax'使用MinMaxScaler
    **kwargs接收对应Scaler支持的参数
    dfs_trans为需要以df_fit为基准进行标准化的pd.DataFrame列表
    返回标准化后的数据以及sklearn的Scaler对象    
    '''
    
    sklScaler_map = {'std': StandardScaler, 'z-score': StandardScaler,
                     'maxmin': MinMaxScaler, 'minmax': MinMaxScaler}
    sklScaler = sklScaler_map[scale_type]
    
    cols_all = list(df_fit.columns)
    
    if cols is None:
        scaler = sklScaler(**kwargs).fit(df_fit)
        df_fited = pd.DataFrame(scaler.transform(df_fit),
                                         columns=cols_all, index=df_fit.index)
        if dfs_trans is None:
            return df_fited, None, (scaler, cols_all)
        dfs_transed = [pd.DataFrame(scaler.transform(df),
                       columns=cols_all, index=df.index) for df in dfs_trans]
        return df_fited, dfs_transed, (scaler, df_fit.columns)
    
    cols_rest = [x for x in cols_all if x not in cols]    
    df_toFit = df_fit.reindex(columns=cols)
    
    scaler = sklScaler(**kwargs).fit(df_toFit)   
    df_fited = pd.DataFrame(scaler.transform(df_toFit), columns=cols,
                                                        index=df_toFit.index)
    for col in cols_rest:
        df_fited[col] = df_fit[col]
    df_fited = df_fited.reindex(columns=cols_all)      
    if dfs_trans is None:
        return df_fited, None, (scaler, cols)    
    dfs_transed = []
    for df in dfs_trans:
        df_trans = df.reindex(columns=cols)
        df_transed = pd.DataFrame(scaler.transform(df_trans),
                                  columns=cols, index=df_trans.index)
        cols_all_df = list(df.columns)
        cols_rest_df = [x for x in cols_all_df if x not in cols]
        for col in cols_rest_df:
            df_transed[col] = df[col]
        dfs_transed.append(df_transed.reindex(columns=cols_all_df))        
    return df_fited, tuple(dfs_transed), (scaler, cols)


def scale_skl_inverse(scaler, dfs_toInv, cols=None, **kwargs):
    '''
    反标准化还原数据
    scaler为fit过的sklearn Scaler类(如StandardScaler、MinMaxScaler)
    dfs_toInv为待还原的df列表
    **kwargs接收scaler.inverse_transform函数支持的参数
    注：(scaler, cols)应与scale_skl函数输出一致
    '''
    
    dfs_inved = []
    
    if cols is None:
        for df in dfs_toInv:
            df_inved = pd.DataFrame(scaler.inverse_transform(df, **kwargs), 
                                    columns=df.columns, index=df.index)
            dfs_inved.append(df_inved)
        return tuple(dfs_inved)
    
    for df in dfs_toInv:
        cols_all = list(df.columns)
        cols_rest = [x for x in cols_all if x not in cols]
        df_inved = pd.DataFrame(scaler.inverse_transform(df[cols], **kwargs),
                                columns=cols, index=df.index)
        for col in cols_rest:
            df_inved[col] = df[col]
        dfs_inved.append(df_inved.reindex(columns=cols_all))
        
    return tuple(dfs_inved)


def norm_std(series, isReverse=False, ddof=1, returnMeanStd=False):
    '''
    z-score标准化，series为pd.Series或np.array，isReverse设置是否反向
    ddof指定计算标准差时是无偏还是有偏的：
        ddof=1时计算的是无偏标准差（样本标准差，分母为n-1），
        ddof=0时计算的是有偏标准差（总体标准差，分母为n）
    （注：pandas的std()默认计算的是无偏标准差，numpy的std()默认计算的是有偏标准差）
    当returnMeanStd为True时同时返回均值和标准差，为False时不返回
    
    注: Z-score适用于series的最大值和最小值未知或有超出取值范围的离群值的情况。
        （一般要求原始数据的分布可以近似为高斯分布，否则效果可能会很糟糕）
    总体标准差和样本标准差参考:
        https://blog.csdn.net/qxqxqzzz/article/details/88663198
    '''
    Smean, Sstd = series.mean(), series.std(ddof=ddof)
    if not isReverse:
        series_new = (series - Smean) / Sstd
    else:
        series_new = (Smean - series) / Sstd
    if not returnMeanStd:
        return series_new
    else:
        return series_new, (Smean, Sstd)


def norm_linear(x, Xmin, Xmax, Nmin=0, Nmax=1, isReverse=False):
    '''
    线性映射：将取值范围在[Xmin, Xmax]内的x映射到取值范围在[Nmin, Nmax]内的xNew
    isReverse设置是否反向，若为True，则映射到[Nmax, Nmin]范围内
    '''
    
    if x < Xmin or x > Xmax:
        raise ValueError('必须满足 Xmin =< x <= Xmax ！')
    if Nmin >= Nmax or Xmin >= Xmax:
        raise ValueError('必须满足 Xmin < Xmax 且 Nmin < Nmax ！')
    
    if isReverse:
        Nmin, Nmax = Nmax, Nmin
    
    xNew = Nmin + (x-Xmin) * (Nmax-Nmin) / (Xmax-Xmin)
    
    return xNew


def norm_mid(x, x_min, x_max, Nmin=0, Nmax=1, x_best=None):
    '''
    中间型（倒V型）指标的正向化线性映射，新值范围为[Nmin, Nmax]
    (指标值既不要太大也不要太小，适当取中间值最好，如水质量评估PH值)
    x_min和x_max为指标可能取到的最小值和最大值
    x_best指定最优值，若不指定则将x_min和x_max的均值当成最优值
    参考：https://zhuanlan.zhihu.com/p/37738503
    '''    
    x_best = (x_min+x_max)/2 if x_best is None else x_best
    if x <= x_min or x >= x_max:
        return Nmin
    elif x > x_min and x < x_best:
        return norm_linear(x, x_min, x_best, Nmin, Nmax)
    elif x < x_max and x >= x_best:
        return norm_linear(x, x_best, x_max, Nmin, Nmax, isReverse=True)


def norm01_mid(x, x_min, x_max, x_best=None):
    '''
    中间型（倒V型）指标的正向化和（线性）01标准化
    (指标值既不要太大也不要太小，适当取中间值最好，如水质量评估PH值)
    x_min和x_max为指标可能取到的最小值和最大值
    x_best指定最优值，若不指定则将x_min和x_max的均值当成最优值
    参考：https://zhuanlan.zhihu.com/p/37738503
    '''    
    x_best = (x_min+x_max)/2 if x_best is None else x_best
    if x <= x_min or x >= x_max:
        return 0
    elif x > x_min and x < x_best:
        return (x - x_min) / (x_best - x_min)    
    elif x < x_max and x >= x_best:
        return (x_max - x) / (x_max - x_best)


def norm_side(x, x_min, x_max, Nmin=0, Nmax=1, x_worst=None, v_outLimit=None):
    '''
    两边型（V型）指标的正向化线性映射，新值范围为[Nmin, Nmax]
    (指标越靠近x_min或越靠近x_max越好，越在中间越不好)
    x_min和x_max为指标可能取到的最小值和最大值
    x_worst指定最差值，若不指定则将x_min和x_max的均值当成最差值
    v_outLimit指定当x超过x_min或x_max界限之后的正向标准化值，不指定则默认为Nmax
    '''
    v_outLimit = Nmax if v_outLimit is None else v_outLimit
    x_worst = (x_min+x_max)/2 if x_worst is None else x_worst
    if x < x_min or x > x_max:
        return v_outLimit
    elif x >= x_min and x < x_worst:
        return norm_linear(x, x_min, x_worst, Nmin, Nmax, isReverse=True)
    elif x <= x_max and x >= x_worst:
        return norm_linear(x, x_worst, x_max, Nmin, Nmax)
    
    
def norm01_side(x, x_min, x_max, x_worst=None, v_outLimit=1):
    '''
    两边型（V型）指标的正向化和（线性）01标准化
    (指标越靠近x_min或越靠近x_max越好，越在中间越不好)
    x_min和x_max为指标可能取到的最小值和最大值
    x_worst指定最差值，若不指定则将x_min和x_max的均值当成最差值
    v_outLimit指定当x超过x_min或x_max界限之后的正向标准化值，不指定则默认为1
    '''
    x_worst = (x_min+x_max)/2 if x_worst is None else x_worst
    if x < x_min or x > x_max:
        return v_outLimit
    elif x >= x_min and x < x_worst:
        return (x_worst - x) / (x_worst - x_min) 
    elif x <= x_max and x >= x_worst:
        return (x - x_worst) / (x_max - x_worst)
    
    
def norm_range(x, x_min, x_max, x_minMin, x_maxMax, Nmin=0, Nmax=1):
    '''
    区间型指标的正向化正向化线性映射，新值范围为[Nmin, Nmax]
    (指标的取值最好落在某一个确定的区间最好，如体温)
    [x_min, x_max]指定指标的最佳稳定区间，[x_minMin, x_maxMax]指定指标的最大容忍区间
    参考：https://zhuanlan.zhihu.com/p/37738503
    '''
    if x >= x_min and x <= x_max:
        return Nmax
    elif x <= x_minMin or x >= x_maxMax:
        return Nmin
    elif x > x_max and x < x_maxMax:
        return norm_linear(x, x_max, x_maxMax, Nmin, Nmax, isReverse=True)
    elif x < x_min and x > x_minMin:
        return norm_linear(x, x_minMin, x_min, Nmin, Nmax)


def norm01_range(x, x_min, x_max, x_minMin, x_maxMax):
    '''
    区间型指标的正向化和（线性）01标准化
    (指标的取值最好落在某一个确定的区间最好，如体温)
    [x_min, x_max]指定指标的最佳稳定区间，[x_minMin, x_maxMax]指定指标的最大容忍区间
    参考：https://zhuanlan.zhihu.com/p/37738503
    '''
    if x >= x_min and x <= x_max:
        return 1
    elif x <= x_minMin or x >= x_maxMax:
        return 0
    elif x > x_max and x < x_maxMax:
        return 1 - (x - x_max) / (x_maxMax - x_max)
    elif x < x_min and x > x_minMin:
        return 1 - (x_min - x) / (x_min - x_minMin)
        
        
def mse(y_true, y_predict):
    '''
    MSE，y_true和y_predict格式为np.array或pd.Series
    注: 要求y_true和y_predict值一一对应
    '''
    if y_predict.shape != y_true.shape:
        raise ValueError('y_true和y_predict的值必须一一对应！')
    yTrue = pd.Series(y_true).reset_index(drop=True)
    yPred = pd.Series(y_predict).reset_index(drop=True)
    return ((yTrue - yPred) ** 2).mean()


def r2(y_true, y_predict, is_linear=False):
    '''
    计算拟合优度R2
    在线性回归情况下（y_predict是由y_true与自变量X进行线性拟合的预测值），有:
        R2 = 1 - SSres / SStot = SSreg / SStot，
        且此时R2与y_true和y_predict相关系数的平方相等
    非线性回归情况下，1 - SSres / SStot != SSreg / SStot，R2与两者相关系数平方也不相等
    可设置is_linear为True和False进行比较验证
    参考：https://blog.csdn.net/wade1203/article/details/98477034
         https://blog.csdn.net/liangyihuai/article/details/88560859
         https://wenku.baidu.com/view/893b22d66bec0975f465e2b8.html
    '''
    if y_predict.shape != y_true.shape:
        raise ValueError('y_true和y_predict的值必须一一对应！')
    yTrue = pd.Series(y_true).reset_index(drop=True)
    yPred = pd.Series(y_predict).reset_index(drop=True)    
    SStot = sum((yTrue - yTrue.mean()) ** 2)
    if not is_linear:
        SSres = sum((yTrue - yPred) ** 2)
        return 1 - SSres / SStot
    else:
        SSreg = sum((yPred - yTrue.mean()) ** 2)
        return SSreg / SStot
    
    
def r2_by_mse(vMSE, y_true):
    '''
    根据MSE和真实值计算拟合优度R2
    参考：https://blog.csdn.net/wade1203/article/details/98477034
         https://blog.csdn.net/liangyihuai/article/details/88560859
    '''
    y_true_var = pd.Series(y_true).var(ddof=0)
    return 1 - vMSE / y_true_var


def r2_deprecated(y_true, y_predict):
    '''
    拟合优度R2计算
    参考：https://blog.csdn.net/wade1203/article/details/98477034
         https://blog.csdn.net/liangyihuai/article/details/88560859
    '''
    vMSE = mse(y_true, y_predict)
    y_true_var = pd.Series(y_true).var(ddof=0)
    return 1 - vMSE / y_true_var
    
    
def rmse(y_true, y_predict):
    '''
    RMSE，y_true和y_predict格式为np.array或pd.Series
    注: 要求y_true和y_predict值一一对应
    '''
    return mse(y_true, y_predict) ** 0.5
    
    
def mae(y_true, y_predict):
    '''
    MAE，y_true和y_predict格式为np.array或pd.Series
    注: 要求y_true和y_predict值一一对应
    '''
    if y_predict.shape != y_true.shape:
        raise ValueError('y_true和y_predict的值必须一一对应！')
    yTrue = pd.Series(y_true).reset_index(drop=True)
    yPred = pd.Series(y_predict).reset_index(drop=True)
    return (abs(yTrue-yPred)).mean()
    
    
def mape(y_true, y_predict):
    '''
    MAPE，y_true和y_predict格式为np.array或pd.Series
    注: 要求y_true和y_predict值一一对应
    注：当y_true存在0值时MAPE不可用
    '''
    if y_predict.shape != y_true.shape:
        raise ValueError('y_true和y_predict的值必须一一对应！')
    yTrue = pd.Series(y_true).reset_index(drop=True)
    yPred = pd.Series(y_predict).reset_index(drop=True)
    return (abs((yPred-yTrue) / yTrue)).mean()
        
        
def smape(y_true, y_predict):
    '''
    SMAPE，y_true和y_predict格式为np.array或pd.Series
    注: 要求y_true和y_predict值一一对应
    注：当y_true和y_predict均为0值时SMAPE不可用
    公式：https://blog.csdn.net/guolindonggld/article/details/87856780
    '''
    if y_predict.shape != y_true.shape:
        raise ValueError('y_true和y_predict的值必须一一对应！')
    yTrue = pd.Series(y_true).reset_index(drop=True)
    yPred = pd.Series(y_predict).reset_index(drop=True)
    return (abs(yPred-yTrue) / ((abs(yPred)+abs(yTrue)) / 2)).mean()


def AVEDEV(arr):
    '''
    AVEDEV函数
    arr为np.array或pd.Series
    
    参考：
    https://support.microsoft.com/en-us/office/
    avedev-function-58fe8d65-2a84-4dc7-8052-f3f87b5c6639?ui=en-us&
    rs=en-us&ad=us
    '''
    
    return (abs(arr-arr.mean())).mean()


def cal_linear_reg_r(y, x=None):
    '''
    计算y中数据点的斜率（一元线性回归）
    y和x为list或pd.Series或np.array
    '''
    if isnull(x):
        X = pd.DataFrame({'X': range(0, len(y))})
    else:
        X = pd.DataFrame({'X': x})
    y = pd.Series(y)
    mdl = lr().fit(X, y)
    return mdl.coef_[0], mdl.intercept_


class External_Std(object):
    '''极端值处理，标准差倍数法'''
    
    def __init__(self, nstd=3, cols=None):
        raise NotImplementedError
    
    # def deal_ext_value(df_fit, cols=None, dfs_trans=None, nstd=5):
    #     '''极端值处理'''
        
    #     cols = df_fit.columns if cols is None else cols
        
    #     mean_stds = []
    #     for col in cols:
    #         mean_stds.append((col, df_fit[col].mean(), df_fit[col].std()))
            
    #     df_fited = df_fit.copy()
    #     for (col, mean, std) in mean_stds:
    #         df_fited[col] = df_fited[col].apply(
    #                         lambda x: np.clip(x, mean-nstd*std, mean+nstd*std))   
    
    #     if dfs_trans is not None:
    #         dfs_traned = []
    #         for df in dfs_trans:
    #             df_tmp = df.copy()
    #             for (col, mean, std) in mean_stds:
    #                 df_tmp[col] = df_tmp[col].apply(
    #                         lambda x: np.clip(x, mean-nstd*std, mean+nstd*std))
    #             dfs_traned.append(df_tmp)
        
    #     return df_fited, tuple(dfs_traned)


def parmsEst():
    '''各种分布不同方法的参数估计'''
    raise NotImplementedError
    
    
def auc():
    raise NotImplementedError


def VarHomTest(s1, s2):
    '''方差齐性检验'''
    raise NotImplementedError


def IndTtest(s1, s2):
    '''独立样本T检验'''
    raise NotImplementedError
    
	
def RelTtest(s1, s2):
    '''配对样本T检验'''
    raise NotImplementedError
    
    
def ANOVA_OneWay(df, col_val, col_group):
    '''单因素方差分析，col_val为待比较列，col_group为组别列'''
    raise NotImplementedError
    