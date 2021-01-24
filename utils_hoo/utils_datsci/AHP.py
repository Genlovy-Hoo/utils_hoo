# -*- coding:utf-8 -*-

import time
import numpy as np
import pandas as pd
from utils_hoo.utils_general import isnull, simple_logger

#%%
def Score2Judge(df_score, max_important_level=9, judge_type='big_score'):
    '''
    将指标重要性评分转化为判断矩阵
    df_score: pd.DataFrame，每列一个指标，每行为一个专家对不同指标的重要性打分，
              打分要求均为正值，越大表示越重要
    max_important_level: 构造判断矩阵时指标两两比较最大等级差，默认9
                        （两两比较重要性等级范围为1/9-9）
    judge_type: 指定计分方式，'big'表示两指标评分比较时，大于就记1分，
                'big_score'表示记分数差值，'score'表示直接用各自得分之和表示重要程度
    返回判断矩阵judge_mat，pd.DataFrame格式
    '''
    
    def importance_cols(colA, colB, judge_type='big_score'):
        '''
        若某个专家对colA列的打分高于colB列的打分，则a_b计分，否则不计分
        a_b的分值代表了colA列相对于colB列的重要性程度，b_a同理
        judge_type: 指定计分方式，'big'表示两指标评分比较时，大于就记1分，
                    'big_score'表示记分数差值，'score'表示直接使用得分
        '''
        df = df_score[[colA, colB]].copy()
        if judge_type == 'score':
            df['a_b'] = df[colA]
            df['b_a'] = df[colB]
        elif judge_type == 'big_score':
            df['a_b'] = df[colA] - df[colB]
            df['b_a'] = df[colB] - df[colA]
        elif judge_type == 'big':
            df['a_b'] = df[colA] - df[colB]
            df['b_a'] = df[colB] - df[colA]
            df['a_b'] = df['a_b'].apply(lambda x: 1 if x > 0 else 0)
            df['b_a'] = df['b_a'].apply(lambda x: 1 if x > 0 else 0)
        else:
            raise ValueError('judge_type应为`big`或`big_score`或`score`！')
        a_b = df[df['a_b'] > 0]['a_b'].sum()
        b_a = df[df['b_a'] > 0]['b_a'].sum()
        return a_b, b_a
    
    judge_mat = pd.DataFrame(np.zeros((df_score.shape[1], df_score.shape[1])))
    judge_mat.columns = df_score.columns
    judge_mat.index = df_score.columns
    for colA in df_score.columns:
        for colB in df_score.columns:
            if colA == colB:
                judge_mat.loc[colA, colB] = 1
            else:
                a_b, b_a = importance_cols(colA, colB, judge_type=judge_type)
                if a_b >= b_a:
                    b_a = 1 if b_a == 0 else b_a
                    v = round(a_b / b_a, 0)
                    v = min(max_important_level, v) # 控制最大值
                    v = max(v, 1) # 控制最小值
                    judge_mat.loc[colA, colB] = v
                    judge_mat.loc[colB, colA] = 1 / v
                    
    return judge_mat

#%%
def checkJM(judge_mat, tol=0.001, logger=None):
    '''
    检查判断矩阵是否符合条件：对称位置乘积是否为1，tol为乘积与1比较时的误差范围控制
    '''
    JM = np.array(judge_mat)
    for i in range(JM.shape[0]):
        for j in range(JM.shape[1]):
            if abs(JM[i, j] * JM[j, i] - 1) > tol:
                if isnull(logger):
                    print(f'判断矩阵可能错误的行和列位置: {i}, {j}')
                else:
                    logger.warning(f'判断矩阵可能错误的行和列位置: {i}, {j}')
                return False
    return True

#%%
def cal_weights(judge_mat, RI_dict=None, checkJMtol=0.001, logger=None):
    '''
    精确方法根据判断矩阵计算权重并进行一致性检验
    判断矩阵judge_mat应为np.array或pd.DataFrame
    RI_dict为随机一致性指标参考值，dict或None
    checkJMtol为checkJM的tol参数
    返回权重向量w以及是否通过一致性检验（True or False）和(CR, CI, lmdmax,RI)等检验信息
    '''
    
    JMOK = checkJM(judge_mat, tol=checkJMtol, logger=logger)
    if not JMOK:
        if isnull(logger):
            print('判断矩阵可能有误，请检查！')
        else:
            logger.warning('判断矩阵可能有误，请检查！')
    
    if RI_dict is None:
        # https://wenku.baidu.com/view/0fa59423336c1eb91b375d32.html
        RI_dict = {1: 0, 2: 0, 3: 0.52, 4: 0.89, 5: 1.12, 6: 1.24, 7: 1.32,
                   8: 1.41, 9: 1.45, 10: 1.49, 11: 1.52, 12: 1.54, 13: 1.56,
                   14: 1.58, 15: 1.59, 16: 1.5943, 17: 1.6064, 18: 1.6133,
                   19: 1.6207, 20: 1.6292, 21: 1.6385, 22: 1.6403, 23: 1.6462,
                   24: 1.6497, 25: 1.6556, 26: 1.6587, 27: 1.6631, 28: 1.667,
                   29: 1.6693, 30: 1.6724}
    num_indicator = judge_mat.shape[0] # 阶数
    
    if num_indicator == 1:
        w = np.array([1.0])
        isOK, CR, CI, lmdmax, RI = True, np.nan, np.nan, np.nan, np.nan
        lambdas, vectors = np.nan, np.nan
        lmdmax = np.nan
        max_vector = np.nan        
    else:
        lambdas, vectors = np.linalg.eig(judge_mat)    
        lmdmax = np.max(lambdas) # 最大特征值
        lmdmax = lmdmax.real
        idx_max = np.argmax(lambdas)
        max_vector = vectors[:,idx_max] # 最大特征值对应特征向量
        w = max_vector / max_vector.sum()
        w = w.real
        
        if num_indicator == 2:
            isOK, CR, CI, lmdmax, RI = True, np.nan, np.nan, np.nan, np.nan            
        else:
            RI = RI_dict[num_indicator]
            CI = (lmdmax-num_indicator) / (num_indicator-1)
            CI = CI.real
            CR = CI / RI
            CR = CR.real            
            if CR > 0.1:
                isOK = False
            else:
                isOK = True
         
    if not isnull(logger):
        logger.info(f'判断矩阵：\n{judge_mat}')
        logger.info(f'特征值：\n{lambdas.real}')
        logger.info(f'特征向量：\n{vectors.real}')
        logger.info(f'最大特征值：{lmdmax.real}')
        logger.info(f'最大特征值对应特向量：\n{max_vector.real}')
        logger.info(f'标准化权重：\n{w}')
        logger.info(f'RI: {RI}')
        logger.info(f'CI: {CI}')
        logger.info(f'CR: {CR}')
    
    return w, isOK, (CR, CI, lmdmax, RI)


def cal_weights_sim(judge_mat, RI_dict=None, checkJMtol=0.001, logger=None):
    '''
    近似方法根据判断矩阵计算权重并进行一致性检验
    判断矩阵judge_mat应为np.array或pd.DataFrame
    RI_dict为随机一致性指标参考值，dict或None
    checkJMtol为checkJM的tol参数
    返回权重向量w以及是否通过一致性检验（True or False）和(CR, CI, lmdmax,RI)等检验信息
    
    参考：
        层次分析：层次分析法在确定绩效指标权重中的应用.pdf
        https://zhuanlan.zhihu.com/p/37738503
    '''
    
    JMOK = checkJM(judge_mat, tol=checkJMtol, logger=logger)
    if not JMOK:
        if isnull(logger):
            print('判断矩阵可能有误，请检查！')
        else:
            logger.warning('判断矩阵可能有误，请检查！')
    
    if RI_dict is None:
        # https://wenku.baidu.com/view/0fa59423336c1eb91b375d32.html
        RI_dict = {1: 0, 2: 0, 3: 0.52, 4: 0.89, 5: 1.12, 6: 1.24, 7: 1.32,
                   8: 1.41, 9: 1.45, 10: 1.49, 11: 1.52, 12: 1.54, 13: 1.56,
                   14: 1.58, 15: 1.59, 16: 1.5943, 17: 1.6064, 18: 1.6133,
                   19: 1.6207, 20: 1.6292, 21: 1.6385, 22: 1.6403, 23: 1.6462,
                   24: 1.6497, 25: 1.6556, 26: 1.6587, 27: 1.6631, 28: 1.667,
                   29: 1.6693, 30: 1.6724}
    num_indicator = judge_mat.shape[0] # 阶数
    
    if num_indicator == 1:
        w = np.array([1.0])
        isOK, CR, CI, lmdmax, RI = True, np.nan, np.nan, np.nan, np.nan
        lmdmax = np.nan
        max_vector = np.nan
    else:        
        sum_cols = judge_mat.sum(axis=0) # 每列求和
        new_jd_mat = judge_mat/sum_cols # 标准化的新矩阵，每列和为1
        max_vector = new_jd_mat.sum(axis=1) # 特征向量
        w = max_vector / num_indicator # 权重向量
        if num_indicator == 2:
            isOK, CR, CI, lmdmax, RI = True, np.nan, np.nan, np.nan, np.nan
        else:
            AW = (w * judge_mat).sum(axis=1)
            lmdmax = sum(AW / (num_indicator*w))
            RI = RI_dict[num_indicator]
            CI = (lmdmax-num_indicator) / (num_indicator-1)
            CR = CI / RI
            if CR > 0.1:
                isOK = False
            else:
                isOK = True
              
    if not isnull(logger):
        logger.info(f'判断矩阵：\n{judge_mat}')
        logger.info(f'最大特征值：{lmdmax}')
        logger.info(f'最大特征值对应特向量：\n{max_vector}')
        logger.info(f'标准化权重：\n{w}')
        logger.info(f'RI: {RI}')
        logger.info(f'CI: {CI}')
        logger.info(f'CR: {CR}')
    
    return w, isOK, (CR, CI, lmdmax, RI)

#%%
def cal_weights_mats(judge_mats, RI_dict=None, WFunc=None, skipBadJM=True,
                     checkJMtol=0.001, logger=None):
    '''
    多个专家（多个判断矩阵）情况下AHP方法计算权重并进行一致性检验
    judge_mats为判断矩阵列表，每个矩阵应为np.array或pd.DataFrame
    RI_dict为随机一致性指标参考值，dict或None
    WFunc指定单个判断矩阵时AHP计算权重的函数，其应接收参数judge_mat、RI_dict、
        checkJMtol和logger，WFunc若不指定，则默认为cal_weights
    返回权重向量w以及是否通过一致性检验（True or False）、
        (CR, CI, lmdmax,RI)等检验信息以及综合判断矩阵和专家判断力权值列表和一致性指标列表
    参考：
        多专家评价的AHP方法及其工程应用
        https://www.doc88.com/p-9913949483932.html
        层次分析法中判断矩阵的群组综合构造方法
        https://www.ixueshu.com/document/1b154ecb637fc9ea05675f42f11aa1a6318947a18e7f9386.html
    '''
    
    if isnull(WFunc):
        WFunc = cal_weights
        
    # 计算每个专家的一致性指标
    us, JMs, CIs, CRs, OKs = [], [], [], [], []
    for k in range(len(judge_mats)):
        w, isOK, (CR, CI, lmdmax, RI) = WFunc(judge_mats[k], RI_dict=RI_dict,
                                        checkJMtol=checkJMtol, logger=None)
        CIs.append(CI)
        CRs.append(CR)
        OKs.append(isOK)
        if isOK:
            us.append(CI)
            JMs.append(judge_mats[k])
        else:
            if skipBadJM:
                if isnull(logger):
                    print(f'第{k+1}个判断矩阵不能通过一致性检验，将被舍弃！')
                else:
                    logger.info(f'第{k+1}个判断矩阵不能通过一致性检验，将被舍弃！')
            else:
                if isnull(logger):
                    print(f'第{k+1}个判断矩阵不能通过一致性检验！')
                else:
                    logger.warning(f'第{k+1}个判断矩阵不能通过一致性检验！')
                us.append(CI)
                JMs.append(judge_mats[k])        
    
    # 计算专家判断力权值
    m = len(us)
    def fP(u):
        return np.exp(-10*(m-1)*u)
    
    P = [fP(u) for u in us]
    P = [x/sum(P) for x in P]
    
    JM = [P[k]*JMs[k] for k in range(m)]
    JM = sum(JM)
    
    w, isOK, (CR, CI, lmdmax, RI) = WFunc(JM, RI_dict=RI_dict,
                                         checkJMtol=checkJMtol, logger=logger)
    
    return w, isOK, (CR, CI, lmdmax, RI), (JM, P, CIs, CRs)

#%%
if __name__ == '__main__':
    strt_tm = time.time()
    
    #%%
    # 来自论文：土地整治的扶贫成效分析及评价——以北川羌族自治县开坪乡土地开发整理项目为例
    # judge_mat = np.array([[1, 3/2, 3/2],
    #                       [2/3, 1, 3/2],
    #                       [2/3, 2/3, 1]])
    judge_mat = np.array([[1, 4, 3, 2, 5],
                          [1/4, 1, 1/2, 1/3, 2],
                          [1/3, 2, 1, 1/2, 2],
                          [1/2, 3, 2, 1, 3],
                          [1/5, 1/2, 1/2, 1/3, 1]])
    logger = None
    # logger = simple_logger()
    
    w1, OK1, info1 = cal_weights(judge_mat, logger=logger)
    print('\n')
    print(f'OK1: {OK1}')
    print(f'w1: \n{w1}')
    
    w2, OK2, info2 = cal_weights_sim(judge_mat, logger=logger)
    print('\n')
    print(f'OK2: {OK2}')
    print(f'w2: \n{w2}')
    
    # 案例2
    data = pd.read_csv('../test/指标重要性专家评分表2.csv', encoding='gbk')
    data = data.set_index('专家')
    indexs = ['经济效益', '社会效益', '生态效益']
    df_score = data.reindex(columns=indexs).dropna(how='any')

    judge_mat = Score2Judge(df_score, max_important_level=9,
                            judge_type='score')
    
    logger = None
    # logger = simple_logger()
    
    w1, OK1, (CR1, CI1, lmd_max1, RI1) = cal_weights(judge_mat, logger=logger)
    print('\n')
    print(f'OK1: {OK1}')
    print(f'w1: \n{w1}')

    w2, OK2, (CR2, CI2, lmd_max2, RI2) = cal_weights_sim(judge_mat,
                                                         logger=logger)
    print('\n')
    print(f'OK2: {OK2}')
    print(f'w2: \n{w2}')
    
    #%%
    judge_mat1 = np.array([[1, 4, 3, 2, 5],
                           [1/4, 1, 1/2, 1/3, 2],
                           [1/3, 2, 1, 1/2, 2],
                           [1/2, 3, 2, 1, 3],
                           [1/5, 1/2, 1/2, 1/3, 1]])
    judge_mat2 = np.array([[1, 3, 3, 2, 4],
                           [1/3, 1, 1/2, 1/3, 2],
                           [1/3, 2, 1, 1/2, 1],
                           [1/2, 3, 2, 1, 3],
                           [1/4, 1/2, 1, 1/3, 1]])
    judge_mats = [pd.DataFrame(judge_mat1), pd.DataFrame(judge_mat2)]
    # judge_mats = [judge_mat1, judge_mat2]
    
    logger = None
    # logger = simple_logger()
    WFunc1 = None
    WFunc2 = cal_weights_sim
    skipBadJM = True
    # skipBadJM = False
    checkJMtol = 0.5
    
    print('\n')
    w1, isOK1, (CR1, CI1, lmdmax1, RI1), (JM1, P1, CIs1, CRs1) = \
        cal_weights_mats(judge_mats, RI_dict=None, WFunc=WFunc1,
                         skipBadJM=skipBadJM, checkJMtol=checkJMtol,
                         logger=logger)
    print('\n')
    print(f'isOK1: {isOK1}')
    print(f'w1: \n{w1}')
    
    w2, isOK2, (CR2, CI2, lmdmax2, RI2), (JM2, P2, CIs2, CRs2) = \
        cal_weights_mats(judge_mats, RI_dict=None, WFunc=WFunc2,
                         skipBadJM=skipBadJM, checkJMtol=checkJMtol,
                         logger=logger)
    print('\n')
    print(f'isOK2: {isOK2}')
    print(f'w1: \n{w2}')
    
    
    # 案例数据来自论文《多专家评价的AHP方法及其工程应用》
    judge_mat1 = np.array([[1, 4, 5, 7, 7, 3],
                           [1/4, 1, 2, 5, 7, 9],
                           [1/5, 1/2, 1, 1, 3, 4],
                           [1/7, 1/5, 1, 1, 5, 7],
                           [1/7, 1/7, 1/3, 1/5, 1, 2],
                           [1/3, 1/9, 1/4, 1/7, 1/2, 1]])
    judge_mat2 = np.array([[1, 5, 5, 9, 3, 1],
                           [1/5, 1, 3, 5, 1, 6],
                           [1/5, 1/3, 1, 4, 2, 5],
                           [1/9, 1/5, 1/4, 1, 6, 7],
                           [1/3, 1, 1/2, 1/6, 1, 2],
                           [1, 1/6, 1/5, 1/7, 1/2, 1]])
    judge_mat3 = np.array([[1, 1, 3, 2, 3, 1],
                           [1, 1, 2, 2, 1, 1],
                           [1/3, 1/2, 1, 3, 3, 5],
                           [1/2, 1/2, 1/3, 1, 3, 2],
                           [1/3, 1, 1/3, 1/3, 1, 3],
                           [1, 1, 1/5, 1/2, 1/3, 1]])
    judge_mat4 = np.array([[1, 2, 9, 3, 1, 1],
                           [1/2, 1, 4, 5, 2, 2],
                           [1/9, 1/4, 1, 1/3, 3, 4],
                           [1/3, 1/5, 3, 1, 3, 1],
                           [1, 1/2, 1/3, 1/3, 1, 2],
                           [1, 1/2, 1/4, 1, 1/2, 1]])
    judge_mat5 = np.array([[1, 2, 1/3, 1/2, 4, 1],
                           [1/2, 1, 2, 1/3, 4, 3],
                           [3, 1/2, 1, 1, 2, 2],
                           [2, 3, 1, 1, 4, 2],
                           [1/4, 1/4, 1/2, 1/4, 1, 5],
                           [1, 1/3, 1/2, 1/2, 1/5, 1]])
    judge_mats = [judge_mat1, judge_mat2, judge_mat3, judge_mat4, judge_mat5]
    
    logger = None
    # logger = simple_logger()
    WFunc1 = None
    WFunc2 = cal_weights_sim
    # skipBadJM = True
    skipBadJM = False
    checkJMtol = 0.001
    w, isOK, (CR, CI, lmdmax, RI), (JM, P, CIs, CRs) = cal_weights_mats(
                    judge_mats, RI_dict=None, WFunc=WFunc1,
                    skipBadJM=skipBadJM, checkJMtol=checkJMtol,logger=logger)
    print('\n')
    print(f'isOK: {isOK}')
    print(f'w: \n{w}')
    
    # 案例数据来自论文《层次分析法中判断矩阵的群组综合构造方法》
    judge_mat1 = np.array([[1, 3, 5, 4, 7],
                           [1/3, 1, 3, 2, 5],
                           [1/5, 1/3, 1, 1/2, 3],
                           [1/4, 1/2, 2, 1, 3],
                           [1/7, 1/5, 1/3, 1/3, 1]])
    judge_mat2 = np.array([[1, 4, 3, 5, 8],
                           [1/4, 1, 4, 3, 6],
                           [1/3, 1/4, 1, 1, 5],
                           [1/5, 1/3, 1, 1, 7],
                           [1/8, 1/6, 1/5, 1/7, 1]])
    judge_mat3 = np.array([[1, 1/2, 3,2, 5],
                           [2, 1, 5, 1, 2],
                           [1/3, 1/5, 1, 2, 1/2],
                           [1/2, 1, 1/2, 1, 5],
                           [1/5, 1/2, 2, 1/5, 1]])
    judge_mat4 = np.array([[1, 3, 5, 2, 6],
                           [1/3, 1, 1, 3, 2],
                           [1/5, 1, 1, 4, 5],
                           [1/2, 1/3, 1/4, 1, 2],
                           [1/6, 1/2, 1/5, 1/2, 1]])
    judge_mat5 = np.array([[1, 2, 6, 3, 3],
                           [1/2, 1, 2, 5, 4],
                           [1/6, 1/2, 1, 1/2, 1],
                           [1/3, 1/5, 2, 1, 5],
                           [1/3, 1/4, 1, 1/5, 1]])
    judge_mat6 = np.array([[1, 2, 5, 4, 9],
                           [1/2, 1, 3, 2, 6],
                           [1/5, 1/3, 1, 1, 2],
                           [1/4, 1/2, 1, 1, 3],
                           [1/9, 1/6, 1/2, 1/3, 1]])
    judge_mats = [judge_mat1, judge_mat2, judge_mat3, judge_mat4, judge_mat5,
                  judge_mat6]
    
    logger = None
    # logger = simple_logger()
    WFunc1 = None
    # WFunc2 = cal_weights_sim
    skipBadJM = True
    # skipBadJM = False
    checkJMtol = 0.1
    
    w, isOK, (CR, CI, lmdmax, RI), (JM, P, CIs, CRs) = cal_weights_mats(
                    judge_mats, RI_dict=None, WFunc=WFunc1,
                    skipBadJM=skipBadJM, checkJMtol=checkJMtol,logger=logger)
    print('\n')
    print(f'isOK: {isOK}')
    print(f'w: \n{w}')
    
    #%%
    print(f'\nused time: {round(time.time()-strt_tm, 6)}s.')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    