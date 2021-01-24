# -*- coding: utf-8 -*-

import time
import random
import numpy as np


def cal_CI(judge_mat):
    '''计算判断矩阵judge_mat的CI值，精确方法'''
    num_indicator = judge_mat.shape[0] # 阶数
    if num_indicator == 1 or num_indicator == 2:
        return 0
    lambdas, _ = np.linalg.eig(judge_mat)
    lambda_max = np.max(lambdas) # 最大特征值
    CI = (lambda_max-num_indicator) / (num_indicator-1)
    return CI


def cal_CI_sim(judge_mat):
    '''计算判断矩阵judge_mat的CI值，近似方法'''
    num_indicator = judge_mat.shape[0] # 阶数
    if num_indicator == 1 or num_indicator == 2:
        return 0
    sum_cols = judge_mat.sum(axis=0) # 每列求和
    new_jd_mat = judge_mat/sum_cols # 标准化的新矩阵，每列和为1
    sum_rows = new_jd_mat.sum(axis=1) # 特征向量
    w = sum_rows / num_indicator # 权重向量
    AW = (w * judge_mat).sum(axis=1)
    lambda_max = sum(AW / (num_indicator*w))
    CI = (lambda_max-num_indicator) / (num_indicator-1)
    return CI


def gen_random_mat(num_indicator, choices):
    '''产生num_indicator阶随机对称矩阵（即判断矩阵），随机值从choices中选择'''
    mat = np.ones((num_indicator, num_indicator))
    for r in range(0, num_indicator):
        for c in range(r+1, num_indicator):
            mat[r,c] = random.choice(choices)
            mat[c,r] = 1 / mat[r,c]
    return mat


def sim_ri(num_indicator, max_important_level=9, num_sim=500):
    '''
    模拟RI值
    num_indicator: 指标个数，即判断矩阵阶数
    max_important_level: 构造判断矩阵时指标两两比较最大等级差，默认9
                        （两两比较重要性等级范围为1/9-9）
    '''
    # choices为判断矩阵可能的取值列表
    choices = list(range(1, max_important_level+1))
    choices = choices + [1/x for x in choices if x != 1]
    # 模拟过程
    CIs = []
    for k in range(num_sim):
        mat = gen_random_mat(num_indicator, choices) # 随机产生判断矩阵
        CI = cal_CI(mat) # 计算CI
        CIs.append(CI)
    RI = np.array(CIs).mean().real # RI为多次模拟的平均值
    return RI


def get_RIs(max_num_indicator=20, num_sim=50000):
    '''
    AHP中随机一致性指标RI的模拟
    max_num_indicator设置模拟的最大指标数量，num_sim设置随机模拟次数
    
    参考：
        https://zhidao.baidu.com/question/13745867.html
        层次分析：层次分析法在确定绩效指标权重中的应用.pdf
        https://zhuanlan.zhihu.com/p/37738503
    '''
    RI_dict={}
    for k in range(1, max_num_indicator+1):
        # max_important_level = 9 if k <= 9 else max_num_indicator
        max_important_level = 9
        RI_dict[k] = sim_ri(k, max_important_level, num_sim)
    return RI_dict


if __name__ == '__main__':
    strt_tm = time.time()
    
    RI_dict = get_RIs()
    
    print(f'used time: {round(time.time()-strt_tm, 6)}s.')
    