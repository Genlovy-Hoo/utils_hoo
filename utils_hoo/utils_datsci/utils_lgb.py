# -*- coding: utf-8 -*-

'''
LightGBM核心参数说明
    （算法类型）
    boosting: 可选['gbdt', 'rf', 'dart', 'goss']（使用dart？可能准确率更高）
    extra_trees: 若为True，则特征分裂时使用极端随机树方式（普通决策树对某个特征进行分裂
    时通过比较不同划分点来选择最优分裂点，极端随机树则随机选择分裂点），使用extra_trees
    可以减少过拟合
    
    （树结构）
    max_depth: 树的深度，太大可能过拟合
    num_leaves: 树的叶子节点数量，太大可能过拟合（二叉树叶子节点数量为: 2^树深度）
    min_data_in_leaf: 每个叶子节点上的样本数，太小可能过拟合
    注：LightGBM使用leaf-wise的算法，因此在调节树的复杂程度时，应使用的是num_leaves，
    而不是max_depth。其值应设置为小于2^(max_depth)，否则容易导致过拟合。
    
    （采样和特征处理）
    bagging_fraction: 样本采样比例，用于降低过拟合和提升训练速度
    bagging_freq: 采样频率，即每个多少轮对样本进行一次采样
    注：bagging_fraction和bagging_freq须同时设置（具体过程？每轮用的样本不一样？）
    feature_fraction: 特征采样比例，即在每轮迭代（每棵树？）随机选取指定比例的特征参与
    建模，用于降低过拟合和提升训练速度
    max_bin: 特征离散化时的分组数（lightgbm采用直方图对特征进行离散化），
    越小训练速度越快，越大越容易过拟合
    
    （正则化）
    lambda_l1: L1正则化项系数
    lambda_l2: L2正则化项系数
    min_gain_to_split: 节点分裂时的最小gain？控制，越大可以减少过拟合
    min_sum_hessian_in_leaf: 每个叶子节点上的hessian矩阵？之和的最小值限制，
    起正则化和限制树深度的作用，越大可以减少过拟合
    path_smooth: 树节点平滑？，越大可以减少过拟合
    注：设置path_smooth大于0时min_data_in_leaf必须大于等于2
    
    （训练过程控制）
    learning_rate, num_iterations: 
        设置较小的learning_rate配合较大的num_iterations比较好
        
其它注意事项：
    categorical_feature参数值在训练函数lgb.train中与在lgb.Dataset中须一致，其对应
    变量必须是大于等于0的整数值，须与数据集中的类别型（category）变量名列相同，最好在
    pandas.DataFrame中设置为.astype('category')
    多分类任务中，标签y的值必须为[0, num_class)范围内的整数，num_class为类别数
    二分类任务中，lgb貌似会默认将y中小于等于0的值当做负标签，大于0的当做正标签
    'num_boost_round'和树的数量是啥关系？
'''

import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from itertools import product
from sklearn.model_selection import KFold
from utils_hoo.utils_io import pickleFile, unpickeFile
from utils_hoo.utils_general import simple_logger, isnull


def check_parms_mdl(parms_mdl, objective, num_class, logger=None):
    '''
    检查模型相关参数设置是否正确
    parms_mdl为设置的模型参数
    检查的参数项目（可添加新的检查项目）：
        任务类型objective
        multiclass中的num_class参数
    返回检查的参数值
    注：后续若更改该函数检查的参数项，则在调用该函数的地方也须做相应修改
    '''
    
    logger = simple_logger() if logger is None else logger
    
    # objective检查
    if objective is None:
        if isinstance(parms_mdl, dict):
            if 'objective' not in parms_mdl.keys():
                raise ValueError('必须设置任务类型objective或在parms_mdl中设置！')
            else:
                objective = parms_mdl['objective']
        else:
            raise ValueError('必须设置任务类型objective或在parms_mdl中设置！')
    else:
        if isinstance(parms_mdl, dict) and 'objective' in parms_mdl.keys() \
                                      and parms_mdl['objective'] != objective:
            logger.warning('objective与parms_mdl设置不一样，以前者为准！')
            parms_mdl['objective'] = objective
            
    if objective not in ['multiclass', 'binary', 'regression']:
        raise ValueError('{}不是允许的任务类型！'.format(objective))
            
    # multiclass的num_class检查
    if objective in ['multiclass']:
        if not isinstance(num_class, int):
            if 'num_class' not in parms_mdl.keys():
                raise ValueError('多分类任务必须指定类别数num_class, int!')
            else:
                num_class = parms_mdl['num_class']
        else:
            if isinstance(parms_mdl, dict) and 'num_class' in \
                     parms_mdl.keys() and num_class != parms_mdl['num_class']:
                logger.warning('num_class与parms_mdl设置不一样，以前者为准！')
                parms_mdl['num_class'] = num_class
                
    return objective, num_class


def get_parms_mdl(parms_mdl=None, objective=None, num_class=None, logger=None):
    '''
    获取模型参数
    parms_mdl为设置的模型参数，若关键参数没设置，则会补充设置默认值
    objective为任务类型，支持的任务类型（可添加其他任务类型）：
        multiclass、binary、regression
    num_class：多分类任务中的类别数（对多分类任务起作用，会对此参数进行特殊检查）
    注：若新增其它任务类型，可能需要设置对应需要特殊检查的参数
    注意：由于lgb参数有别称，可能导致混淆或重复设置，故parms_mdl中出现的参数名称应与
    本函数中默认名称保持一致！
    '''
    
    logger = simple_logger() if logger is None else logger
    
    objective, num_class = check_parms_mdl(parms_mdl, objective, num_class,
                                           logger=logger)
    
    # 损失函数
    if objective == 'multiclass':
        metric = ['multi_logloss', 'multi_error']
    elif objective == 'binary':
        metric = ['binary_logloss', 'auc', 'binary_error']
    elif objective == 'regression':
        metric = ['l1', 'l2', 'mape'] # l1=mae, l2=mse
    
    # 默认参数
    parms_mdl_must_default = {'objective': objective,
                              'num_class': num_class,
                              'metric': metric,
                              'boosting': 'gbdt',
                              'extra_trees': False,
                              
                              'max_depth': 3,
                              'num_leaves': 31,
                              'min_data_in_leaf': 20,
                              
                              'bagging_fraction': 0.75,
                              'bagging_freq': 5,
                              'feature_fraction': 0.75,
                              'max_bin': 255,
                              
                              'lambda_l1': 0.1,
                              'lambda_l2': 0.1,
                              'min_gain_to_split': 0.01,
                              'min_sum_hessian_in_leaf': 0.01,
                              'path_smooth': 0.0,
                              
                              'learning_rate': 0.05,
                              'is_unbalance': False,
                              # 'random_state': 62,
                              'random_state': None,
                              
                              'num_threads': 4} 
    
    if parms_mdl is None:
        parms_mdl = parms_mdl_must_default
    parms_mdl_loss = {x: parms_mdl_must_default[x] \
                                  for x in parms_mdl_must_default.keys() if \
                                      x not in parms_mdl.keys()}
    parms_mdl.update(parms_mdl_loss)
    
    return parms_mdl


def get_parms_TrainOrCV(parms_TrainOrCV=None):
    '''
    获取lgb.train或lgb.cv参数
    parms_TrainOrCV为以字典形式设置的train或cv参数，
    若关键参数没设置，则会补充设置默认值
    '''
    
    # lgb.train或lgb.cv必要的默认参数
    parms_default = {'num_boost_round': 1000,
                     'fobj': None, # ？
                     'feval': None, # ？
                     'init_model': None,
                     'feature_name': 'auto',
                     'categorical_feature': 'auto',
                     'early_stopping_rounds': 100,
                     'verbose_eval': 50,
                     'callbacks': None, # ？
                     
                     # 仅lgb.train调用参数                    
                     'learning_rates': None,
                     'keep_training_booster': False,
                     
                     # 仅lgb.cv调用参数
                     'folds': None,
                     'nfold': 5,
                     'stratified': True, # 分层抽样
                     'shuffle': True,
                     'metrics': None, # 会覆盖模型参数中的metric
                     'fpreproc': None, # ？
                     'show_stdv': True, # ？
                     # 'seed': 62, # 交叉验证生成样本时的随机数种子
                     'seed': None,
                     'eval_train_metric': False,
                     'return_cvbooster': True}
    
    if parms_TrainOrCV is None:
        parms_TrainOrCV = parms_default
    parms_loss = {x: parms_default[x] \
                                for x in parms_default.keys() if \
                                    x not in parms_TrainOrCV.keys()}
    parms_TrainOrCV.update(parms_loss)
    
    return parms_TrainOrCV


def lgb_train(X_train, y_train, X_valid=None, y_valid=None, objective=None,
              parms_mdl=None, parms_train=None, mdl_save_path=None, 
              logger=None):
    '''
    lightgbm模型训练
    X_train, y_train, X_valid, y_valid为pd或np格式，最好为pd格式数据
    objective为任务类型，支持的任务类型（可添加其他任务类型）：
        multiclass、binary、regression
    parms_mdl和parms_train为模型参数和训练参数（dict）
    mdl_save_path为模型本地化路径
    返回训练好的模型和损失函数变化曲线数据
    '''
    
    logger = simple_logger() if logger is None else logger
    
    # 检查任务相关参数（注意：若添加其他任务，可能需要添加对应需要检查的参数）
    num_class = len(set(y_train)) if objective == 'multiclass' else 1
    objective, num_class = check_parms_mdl(parms_mdl, objective, num_class,
                                           logger=logger) 
    # 模型参数和训练参数准备
    parms_mdl = get_parms_mdl(parms_mdl=parms_mdl, objective=objective,
                              num_class=num_class, logger=logger)
    parms_train = get_parms_TrainOrCV(parms_TrainOrCV=parms_train)
        
    # 数据集准备
    datTrain = lgb.Dataset(X_train, y_train,
                    categorical_feature=parms_train['categorical_feature'])
    if X_valid is None and y_valid is None:
        datValid = None
    else:
        datValid = lgb.Dataset(X_valid, y_valid,
                    categorical_feature=parms_train['categorical_feature'])
    valid_sets = [datTrain, datValid] if datValid is not None else [datTrain]
    valid_names = ['train', 'valid'] if X_valid is not None else ['train']    
    evals_result = {}
        
    # 模型训练
    logger.info('模型训练中...')
    mdl = lgb.train(params=parms_mdl, 
                    train_set=datTrain, 
                    num_boost_round=parms_train['num_boost_round'],
                    valid_sets=valid_sets, 
                    valid_names=valid_names,
                    fobj=parms_train['fobj'],
                    feval=parms_train['feval'],
                    init_model=parms_train['init_model'],
                    feature_name=parms_train['feature_name'],
                    categorical_feature=parms_train['categorical_feature'],
                    early_stopping_rounds=parms_train['early_stopping_rounds'],
                    evals_result=evals_result,
                    verbose_eval=parms_train['verbose_eval'],                    
                    learning_rates=parms_train['learning_rates'],
                    keep_training_booster=parms_train['keep_training_booster'],
                    callbacks=parms_train['callbacks'])
    
    # 模型保存
    if not isnull(mdl_save_path):
        # joblib.dump(mdl, 'mdl_save_path')
        pickleFile(mdl, 'mdl_save_path')
        
    return mdl, evals_result


def lgb_cv(X_train, y_train, objective=None, parms_mdl=None, parms_cv=None,
           logger=None):
    '''
    lightgbm交叉验证
    X_train, y_train为pd或np格式，最好为pd格式数据
    objective为任务类型，支持的任务类型（可添加其他任务类型）：
        multiclass、binary、regression
    parms_mdl和parms_cv为模型参数和cv参数（dict）
    返回各损失函数值？
    '''
    
    logger = simple_logger() if logger is None else logger
    
    # 检查任务相关参数（注意：若添加其他任务，可能需要添加对应需要检查的参数）
    num_class = len(set(y_train)) if objective == 'multiclass' else 1
    objective, num_class = check_parms_mdl(parms_mdl, objective, num_class,
                                           logger=logger) 
    # 模型参数和训练参数准备    
    parms_mdl = get_parms_mdl(parms_mdl=parms_mdl, objective=objective,
                              num_class=num_class, logger=logger)
    parms_cv = get_parms_TrainOrCV(parms_TrainOrCV=parms_cv)
    if objective == 'regression':
        parms_cv['stratified'] = None # 回归任务不适用分层抽样
    
    # 数据集准备
    datTrain = lgb.Dataset(X_train, y_train,
                    categorical_feature=parms_cv['categorical_feature'])
    
    # 交叉验证
    logger.info('交叉验证...')
    eval_hist = lgb.cv(params=parms_mdl,
                       train_set=datTrain,
                       num_boost_round=parms_cv['num_boost_round'],
                       folds=parms_cv['folds'],
                       nfold=parms_cv['nfold'],
                       stratified=parms_cv['stratified'],
                       shuffle=parms_cv['shuffle'],
                       metrics=parms_cv['metrics'],
                       fobj=parms_cv['fobj'],
                       feval=parms_cv['feval'],
                       init_model=parms_cv['init_model'],
                       feature_name=parms_cv['feature_name'],
                       categorical_feature=parms_cv['categorical_feature'],
                       early_stopping_rounds=parms_cv['early_stopping_rounds'],
                       fpreproc=parms_cv['fpreproc'],
                       verbose_eval=parms_cv['verbose_eval'],
                       show_stdv=parms_cv['show_stdv'],
                       seed=parms_cv['seed'],
                       callbacks=parms_cv['callbacks'],
                       eval_train_metric=parms_cv['eval_train_metric'],
                       # return_cvbooster=parms_cv['return_cvbooster']
                       )
    
    return eval_hist


def lgb_cv_GridSearch(X_train, y_train, objective=None, parms_mdl=None,
                      parms_to_opt=None, parms_cv=None, logger=None):
    '''lgb交叉验证网格搜索调参'''
    
    logger = simple_logger() if logger is None else logger
    
    if not isinstance(parms_to_opt, dict) or len(parms_to_opt) == 0:
        logger.error('检测到待优化参数parms_to_opt为空！')
        return None, None
    
    # 检查任务相关参数（注意：若添加其他任务，可能需要添加对应需要检查的参数）
    num_class = len(set(y_train)) if objective == 'multiclass' else 1
    objective, num_class = check_parms_mdl(parms_mdl, objective, num_class,
                                           logger=logger) 
    # 模型参数和训练参数准备    
    parms_mdl = get_parms_mdl(parms_mdl=parms_mdl, objective=objective,
                              num_class=num_class, logger=logger)
    # 注意：这里metric没考虑自定义的情况，自定义metric需要再修改
    if len(parms_mdl['metric']) > 1 and \
                                    not isinstance(parms_mdl['metric'], str):
        metric = list(parms_mdl['metric'])[0]
        logger.warning('发现多个metric，将以{}为优化目标！'.format(metric))
        # 注：由于set是无序的，故当parms_mdl['metric']是set时可能取不到第一个
        parms_mdl['metric'] = metric
    if not isinstance(parms_mdl['metric'], str):
        parms_mdl['metric'] = list(parms_mdl['metric'])[0]
        
    # 判断metric越大越好还是越小越好
    if parms_mdl['metric'] in ['auc']:
        max_good = True
        best = -np.inf
    elif parms_mdl['metric'] in ['l1', 'l2', 'mape', 'rmse', 'binary_error',
                                 'multi_error', 'binary_logloss',
                                 'multi_logloss']:
        max_good = False
        best = np.inf
    else:
        raise ValueError('未识别的metric: {}，请更改或在此函数中增加该支持项！' \
                         .format(parms_mdl['metric']))
    
    # 将待优化参数网格化
    grid_parms = []
    opt_items = sorted(parms_to_opt.items())
    keys, values = zip(*opt_items)
    for v in product(*values):
        grid_parm = dict(zip(keys, v))
        grid_parms.append(grid_parm)
    
    # cv网格搜索
    metric = parms_mdl['metric']
    best_parms = None
    k = 1
    for grid_parm in grid_parms:
        logger.info('交叉验证网格搜索中：{} / {} ...'.format(k, len(grid_parms)))
        logger.info('当前参数：{}'.format(grid_parm))
        k += 1
        
        parms_mdl_now = parms_mdl.copy()
        parms_mdl_now.update(grid_parm)
        
        eval_hist = lgb_cv(X_train, y_train, objective=objective,
                           parms_mdl=parms_mdl_now, parms_cv=parms_cv,
                           logger=logger)
        if max_good:
            best_now = max(eval_hist[metric+'-mean'])
            if best_now > best:
                best = best_now
                best_parms = grid_parm
        else:
            best_now = min(eval_hist[metric+'-mean'])
            if best_now < best:
                best = best_now
                best_parms = grid_parm
    
    return best_parms, {'best '+metric: best}


def lgb_cv_hoo(X_train, y_train, objective=None,
               parms_mdl_list=None, parms_train_list=None,
               Nfold=5, shuffle=True, random_state=62, mdl_path_list=None,
               logger=None):
    '''
    自定义lgb交叉验证，返回模型列表和结果列表
    '''
    
    logger = simple_logger() if logger is None else logger
    
    # 模型参数设置为列表（每个模型单独设置）
    if parms_mdl_list is None or isinstance(parms_mdl_list, dict):
        parms_mdl_list = [parms_mdl_list] * Nfold
        
    # 训练参数设置为列表（每个模型单独设置）
    if parms_train_list is None or isinstance(parms_train_list, dict):
        parms_train_list = [parms_train_list] * Nfold
        
    # 若模型存放路径为str，则新建文件夹并设置保存路径列表
    if mdl_path_list is None:
        mdl_path_list = [None] * Nfold
    elif isinstance(mdl_path_list, str):
        abs_dir = os.path.abspath(mdl_path_list)
        if not os.path.isdir(mdl_path_list):
            logger.warning('将创建模型存放文件夹{}！'.format(abs_dir))
            os.mkdir(abs_dir)
        mdl_path_list = [os.path.join(abs_dir, 'mdl_kf'+str(k)+'.bin') \
                         for k in range(1, Nfold+1)]
    
    # 交叉验证
    mdls, evals_results = [], []
    folds = KFold(n_splits=Nfold, shuffle=shuffle, random_state=random_state)
    for Ikf, (trnIdxs, valIdxs) in enumerate(folds.split(X_train, y_train)):
        logger.info('{}/{}折交叉验证训练中...'.format(Ikf+1, Nfold))
        
        if isinstance(X_train, pd.core.frame.DataFrame):
            X_train_Ikf = X_train.iloc[trnIdxs, :]
            X_valid_Ikf = X_train.iloc[valIdxs, :]
        elif isinstance(X_train, np.ndarray):
            X_train_Ikf = X_train[trnIdxs]
            X_valid_Ikf = X_train[valIdxs]
            
        if isinstance(y_train, pd.core.series.Series):
            y_train_Ikf = y_train.iloc[trnIdxs]
            y_valid_Ikf = y_train.iloc[valIdxs]
        elif isinstance(y_train, np.ndarray):
            y_train_Ikf = y_train[trnIdxs]
            y_valid_Ikf = y_train[valIdxs]
            
        mdl, evals_result = lgb_train(X_train_Ikf, y_train_Ikf,
                                      X_valid=X_valid_Ikf, y_valid=y_valid_Ikf,
                                      objective=objective,
                                      parms_mdl=parms_mdl_list[Ikf],
                                      parms_train=parms_train_list[Ikf],
                                      mdl_save_path=mdl_path_list[Ikf],
                                      logger=logger)
        mdls.append(mdl)
        evals_results.append(evals_result)
        
    return mdls, evals_results
    

def lgb_predict(mdl, X, p_cut=0.5):
    '''
    lgb模型预测
    mdl为训练好的模型或本地模型路径，X为待预测输入
    p_cut为二分类确定标签时的概率切分点
    '''
    
    # 模型导入
    if isinstance(mdl, str):
        if os.path.exists(mdl):
            # mdl = joblib.load(mdl)
            mdl = unpickeFile(mdl)
        else:
            raise ValueError('检测到输入mdl可能为路径，但是模型文件不存在！')
         
    # 是否有最优迭代次数
    best_itr = mdl.best_iteration if mdl.best_iteration > 0 else None
    mdl_output = mdl.predict(X, num_iteration=best_itr)
    
    # 回归预测
    if mdl.params['objective'] in ['regression']:
        return mdl_output, None
    
    # 多分类预测
    elif mdl.params['objective'] in ['multiclass']:
        return mdl_output.argmax(axis=1), mdl_output
    
    # 二分类预测
    elif mdl.params['objective'] in ['binary']:
        label_pre = np.zeros(mdl_output.shape)
        label_pre[np.where(mdl_output >= p_cut)] = 1
        return label_pre, mdl_output
    
    
def get_features_importance(mdl, sum1=True, return_type='df'):
    '''
    返回特征重要性
    sum1设置是否标准化成和为1的权重
    return_type可选[`df`, `dict`]'''
    fimp = pd.DataFrame(mdl.feature_importance())
    fimp.index = mdl.feature_name()
    fimp.columns = ['importance']
    fimp.sort_values('importance', ascending=False, inplace=True)
    fimp.index.name = 'feature_name'
    if sum1:
        fimp['importance'] = fimp['importance'] / fimp['importance'].sum()
    if return_type == 'df':
        return fimp
    elif return_type == 'dict':
        return fimp['importance'].to_dict()






