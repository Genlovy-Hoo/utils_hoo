# -*- coding: utf-8 -*-

import time
import pandas as pd
from utils_hoo.utils_optimizer.GA import GA
from utils_hoo.utils_optimizer.CS import CS
from utils_hoo.utils_optimizer.PSO import PSO
from utils_hoo.utils_optimizer.GWO import GWO
from utils_hoo.utils_optimizer.WOA import WOA
from utils_hoo.utils_optimizer.HHO import HHO
from utils_hoo.utils_general import simple_logger
from utils_hoo.utils_plot.plot_Common import plot_Series
from utils_hoo.utils_logging.logger_general import get_logger
from utils_hoo.utils_logging.logger_utils import close_log_file
from utils_hoo.utils_optimizer.utils_Heuristic import FuncOpterInfo

from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn import datasets
from sklearn.model_selection import train_test_split as tts
from utils_hoo.utils_datsci.utils_stats import scale_skl, mape
from sklearn import metrics

#%%
def gbc_objf(superParams, Xtrain=None, Ytrain=None, Xtest=None, Ytest=None):
    '''
    构造gbdt分类模型目标函数（适应度函数）
    '''
    max_depth = int(superParams[0])
    subsample = superParams[1]
    min_samples_leaf = int(superParams[2])
    min_samples_split = int(superParams[3])
    mdl = GBC(max_depth=max_depth,
              subsample=subsample,
              min_samples_leaf=min_samples_leaf,
              min_samples_split=min_samples_split)
    mdl = mdl.fit(Xtrain, Ytrain)
    Ypre = mdl.predict(Xtest)
    error = 1 - metrics.accuracy_score(Ytest, Ypre)
    return error
    

def gbr_objf(superParams, Xtrain=None, Ytrain=None, Xtest=None, Ytest=None,
             **kwargs):
    '''
    构造gbdt回归模型目标函数（适应度函数）
    '''
    max_depth = int(superParams[0])
    subsample = superParams[1]
    min_samples_leaf = int(superParams[2])
    min_samples_split = int(superParams[3])
    mdl = GBC(max_depth=max_depth,
              subsample=subsample,
              min_samples_leaf=min_samples_leaf,
              min_samples_split=min_samples_split, **kwargs)
    mdl = mdl.fit(Xtrain, Ytrain)
    Ypre = mdl.predict(Xtest)
    vMAPE = mape(Ytest, Ypre)
    return vMAPE

#%%
if __name__ == '__main__':
    strt_tm = time.time()
    
    #%%
    # 分类任务数据集
    data_cls = datasets.load_iris()
    X_cls = pd.DataFrame(data_cls['data'], columns=data_cls.feature_names)
    Y_cls = pd.Series(data_cls['target'])
   
    Xtrain_cls, Xtest_cls, Ytrain_cls, Ytest_cls = tts(X_cls, Y_cls,
                                        test_size=0.4, random_state=5262)
    Xtrain_cls, [Xtest_cls], _ = scale_skl(Xtrain_cls, [Xtest_cls])
    
    
    # 回归任务数据集
    # data_reg = datasets.load_boston()
    data_reg = datasets.load_diabetes()
    X_reg = pd.DataFrame(data_reg['data'], columns=data_reg.feature_names)
    Y_reg = pd.Series(data_reg['target'])
    
    Xtrain_reg, Xtest_reg, Ytrain_reg, Ytest_reg = tts(X_reg, Y_reg,
                                        test_size=0.4, random_state=5262)
    Xtrain_reg, [Xtest_reg], _ = scale_skl(Xtrain_reg, [Xtest_reg])
    
    #%%
    # 目标函数和参数
    objf = gbr_objf
    parms_func = {'func_name': objf.__name__,
                  'x_lb': [1, 0.01, 1, 2], 'x_ub': [10, 1.0, 10, 10], 'dim': 4,
                  'kwargs': {'Xtrain': Xtrain_reg, 'Ytrain': Ytrain_reg,
                             'Xtest': Xtest_reg, 'Ytest': Ytest_reg,
                             'n_estimators': 100,}}
    # objf = gbc_objf
    # parms_func = {'func_name': objf.__name__,
    #               'x_lb': 0.01, 'x_ub': 100, 'dim': 2,
    #               'kwargs': {'Xtrain': Xtrain_cls, 'Ytrain': Ytrain_cls,
    #                           'Xtest': Xtest_cls, 'Ytest': Ytest_cls}}
    
    # 统一参数
    PopSize = 5
    Niter = 10
    
    # logger
    # logger = simple_logger()
    logger = get_logger('./test/HeuristicSVM.txt', screen_show=True)
    # parms_log = {'logger': logger, 'nshow': 10}
    parms_log = {'logger': logger, 'nshow': Niter}
    
    fvals = pd.DataFrame()
    
    #%%
    # GA
    parms_ga = {'opter_name': 'GA',
                'PopSize': PopSize, 'Niter': Niter,
                'Pcrs': 0.7, 'Pmut': 0.1, 'Ntop': 2}
    
    ga_parms = FuncOpterInfo(parms_func, parms_ga, parms_log)
    ga_parms = GA(objf, ga_parms)
    fvals['GA'] = ga_parms.convergence_curve
    
    # PSO
    parms_pso = {'opter_name': 'PSO',
                 'PopSize': PopSize, 'Niter': Niter,
                 'v_maxs': 5, 'w_max': 0.9, 'w_min': 0.2, 'w_fix': False,
                 'c1': 2, 'c2': 2}
    
    pso_parms = FuncOpterInfo(parms_func, parms_pso, parms_log)
    pso_parms = PSO(objf, pso_parms)
    fvals['PSO'] = pso_parms.convergence_curve
    
    # CS
    parms_cs = {'opter_name': 'CS',
                'PopSize': PopSize, 'Niter': Niter,
                'pa': 0.25, 'beta': 1.5, 'alpha': 0.01}
    
    cs_parms = FuncOpterInfo(parms_func, parms_cs, parms_log)
    cs_parms = CS(objf, cs_parms)
    fvals['CS'] = cs_parms.convergence_curve
    
    # GWO
    parms_gwo = {'opter_name': 'GWO',
                  'PopSize': PopSize, 'Niter': Niter}
    
    gwo_parms = FuncOpterInfo(parms_func, parms_gwo, parms_log)
    gwo_parms = GWO(objf, gwo_parms)
    fvals['GWO'] = gwo_parms.convergence_curve
    
    # WOA
    parms_woa = {'opter_name': 'WOA',
                  'PopSize': PopSize, 'Niter': Niter}
    
    woa_parms = FuncOpterInfo(parms_func, parms_woa, parms_log)
    woa_parms = WOA(objf, woa_parms)
    fvals['WOA'] = woa_parms.convergence_curve
    
    # HHO
    parms_hho = {'opter_name': 'HHO',
                  'PopSize': PopSize, 'Niter': Niter,
                  'beta': 1.5, 'alpha': 0.01}
    
    hho_parms = FuncOpterInfo(parms_func, parms_hho, parms_log)
    hho_parms = HHO(objf, hho_parms)
    fvals['HHO'] = hho_parms.convergence_curve
    
    #%%
    # 作图比较
    plot_Series(fvals.iloc[:, :],
                {'GA': '-', 'PSO': '-', 'CS': '-', 'GWO': '-', 'WOA': '-',
                  'HHO': '-'},
                figsize=(10, 6))
    
    #%%
    close_log_file(logger)
    
    #%%
    print(f'used time: {round(time.time()-strt_tm, 6)}s.')
