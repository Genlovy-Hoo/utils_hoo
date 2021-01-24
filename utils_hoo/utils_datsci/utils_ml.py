# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score

import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['font.serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt


def vote_label_int(label_pred_list):
    '''
    投票法生成最终二分类或多分类label，label_pred_list为不同的预测结果列表，
    每个预测结果是值为整数的np.array或pd.Series或list
    '''
    all_preds = list(zip(*label_pred_list))
    return stats.mode(all_preds, axis=1)[0].reshape((-1,))


def vote_prob_multi(prob_pred_list):
    '''
    投票法生成最终多分类label，prob_pred_list为不同预测结果列表，
    每个预测结果是值为概率的np.array或pd.Series，shape为样本数*类别数
    '''
    probs_sum = sum(prob_pred_list)
    return probs_sum.argmax(axis=1)


def vote_prob_bin_pcut(prob_pred_list, pcut=0.5):
    '''
    投票法生成最终二分类label，prob_pred_list为不同预测结果列表，
    每个预测结果是值为概率（label为1的概率）的np.array或pd.Series，shape为样本数*1
    pcut: 指定将预测概率大于pcut的样本标注为1
    '''
    probs_sum = sum(prob_pred_list)
    label = np.zeros(probs_sum.shape)
    label[np.where(probs_sum >= pcut * len(prob_pred_list))] = 1
    return label


def vote_prob_bin_rate1(prob_pred_list, rate1=0.5):
    '''
    投票法生成最终二分类label，prob_pred_list为不同预测结果列表，
    每个预测结果是值为概率（label为1的概率）的np.array或pd.Series，shape为样本数
    rate1: 指定标签为1的样本比例
    '''
    probs_sum = sum(prob_pred_list)
    label = np.zeros(probs_sum.shape)
    label[np.where(probs_sum >= np.quantile(probs_sum, 1-rate1))] = 1
    return label


def auc_bin(y_true, pred_prob, **kwargs):
    '''
    二分类AUC计算
    y_true为真实标签，pred_prob为预测概率（取y_true中较大值的概率）
    **kwargs可设置roc_auc_score函数接收的其它参数
    '''
    return roc_auc_score(y_true, pred_prob, **kwargs)


def F1Score(y_true, y_pred, **kwargs):
    '''
    F1-score计算
    y_true为真实标签，y_pred为预测标签
    **kwargs可设置f1_score函数接收的其它参数
    '''
    return f1_score(y_true, y_pred, **kwargs)


def ConfusionMatrix(y_true, y_pred):
    '''混淆矩阵，y_true和y_predict分别为真实标签和预测标签'''
    
    real_pre = pd.DataFrame({'真实': y_true, '预测': y_pred})
    cros_tab = pd.crosstab(real_pre['真实'],
                           real_pre['预测'], margins=True)

    cros_tab = cros_tab.reindex(columns=list(cros_tab.index)).fillna(0)
    cros_tab = cros_tab.astype(int)
    
    cros_tab['判断正确'] = 0
    labels = list(set(y_true))
    for label in labels:
        cros_tab.loc[label, '判断正确'] = cros_tab.loc[label, label]    
    cros_tab.loc['All', '判断正确'] = cros_tab['判断正确'].sum()
    
    cros_tab['正确率'] = cros_tab['判断正确'] / cros_tab['All']
    
    return cros_tab
    

def plot_roc_bin(y_true_pred_prob, labels=None, lnstyls=None,
                 figsize=(8, 8), title=None, fontsize=15, grid=False,
                 fig_save_path=None, **kwargs):
    '''
    二分类ROC曲线绘制
    y_true_pred_probs格式：
        [(y_true1, pred_prob1), (y_true2, pred_prob2), ...]
        其中y_true为真实标签
        pred_prob为预测概率结果（取y_true中较大值的概率）
    labels设置每个结果ROC曲线图例标签，list或tuple
    lnstyls设置每个结果ROC曲线的线型，list或tuple
    若只有一个结果，则pred_probs，labels，lnstyls可不用写成list或tuple
    **kwargs可设置roc_curve函数接收的其它参数
    返回AUC值
    '''
    
    # y_true_pred_prob格式检查
    if (not isinstance(y_true_pred_prob, list) and \
        not isinstance(y_true_pred_prob, tuple)) or \
       len(y_true_pred_prob[0]) != 2:
        raise ValueError(
              '须将y_true_pred_prob组织成[(y_true, pred_prob), ...]！')
    
    # 图例标签组织成列表
    if labels is None:
        if len(y_true_pred_prob) == 1:
            labels = ['ROC']
        else:
            labels = ['ROC'+str(k+1) for k in range(0, len(y_true_pred_prob))]
    elif not isinstance(labels, list) and not isinstance(labels, tuple):
        labels = [labels]
    
    # 线型组织成列表
    if lnstyls is None:
        lnstyls = [None] * len(y_true_pred_prob)
    elif not isinstance(lnstyls, list) and \
                                          not isinstance(lnstyls, tuple):
        lnstyls = [lnstyls]
    
    # 计算ROC曲线和AUC
    fpr_tprs, vAUCs = [], []
    for y_true, pred_prob in y_true_pred_prob:
        fpr, tpr, thresholds = roc_curve(y_true, pred_prob,
                                         pos_label=max(y_true), **kwargs)
        vAUC = auc(fpr, tpr)
        fpr_tprs.append((fpr, tpr))
        vAUCs.append(vAUC)
        
    plt.figure(figsize=figsize)
    
    for k in range(len(fpr_tprs)):
        fpr, tpr = fpr_tprs[k]
        AUCstr = str(round(vAUCs[k], 4))
        lbl_str = str(labels[k])
        lnstyl = lnstyls[k]
        if lnstyl is None:
            plt.plot(fpr, tpr, label=lbl_str+'(AUC: '+AUCstr+')')
        else:
            plt.plot(fpr, tpr, lnstyl, label=lbl_str+'(AUC: '+AUCstr+')')
        
    # 对角线
    plt.plot(np.array([0.0, 1.0]), np.array([0.0, 1.0]), 'r-')
        
    plt.legend(loc=0, fontsize=fontsize)
    
    plt.grid(grid)
    
    if title:
        plt.title(title, fontsize=fontsize) 
        
    if fig_save_path:
        plt.savefig(fig_save_path)
        
    plt.show()
    
    if len(y_true_pred_prob) == 1:
        return vAUCs[0]
    
    return vAUCs

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
