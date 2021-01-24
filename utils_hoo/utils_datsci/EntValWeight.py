# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
 

def EntValWeight(df, neg_cols=[], score_type=None):
    '''
    熵值法计算变量的权重
    
    Args:
        df: pd.DataFrame，每行一个样本，每列一个指标，为每个指标计算权重
            df应先删除或填充无效值
        neg_cols: 负向指标列名列表
        score_type: 可选[None, 'ori', 'std']，分别表示不计算每个样本得分、
                    以原始数据计算每个样本得分、以标准化数据计算每个样本得分
                    
    Return:
        w: 以pd.DataFrame保存的权重向量'weight'，w.index为df.columns
        score: 每个样本得分，列名为'score'，若score_type为None，则为None
    
    参考：
    https://www.jianshu.com/p/3e08e6f6e244
    https://blog.csdn.net/qq_24975309/article/details/82026022
    https://wenku.baidu.com/view/06f7590602768e9950e7386a.html
    '''
    
    if score_type not in [None, 'std', 'ori']:
        raise ValueError('score_type必须为None或`std`或`ori`！')    
    cols = list(df.columns)
    
    # 数据标准化（默认0-1标准化，也可以用其它标准化方法？）
    P = df.copy()
    for col in cols:
        if col not in neg_cols:
            P[col] = (P[col]-P[col].min()) / (P[col].max()-P[col].min())
        else:
            P[col] = (P[col].max()-P[col]) / (P[col].max()-P[col].min())
    
    # 每个样本在每列（指标）中的比重
    for col in cols:
        P[col] = P[col] / P[col].sum()
        
    lnP = P.copy()
    for col in cols:
        tol = 1e-6
        lnP[col] = lnP[col].apply(lambda x: np.log(x) if x != 0 else tol)  
    
    k = -1.0 / np.log(df.shape[0]) # 注: k = -1.0 / np.log(df.shape[0]+1)亦可？
    e = k * P * lnP 
    e = e.sum() # 每列（指标）的熵值
    d = 1 - e # 信息熵冗余度
    w = d / d.sum() # 权重向量
    
    w = pd.DataFrame(w)
    w.columns = ['weight']
    
    if score_type is not None:
        df_score = df if score_type == 'ori' else P
        w_rep = [w.transpose()] * df.shape[0]
        w_rep = pd.concat(w_rep, axis=0)
        w_rep.index = df.index
        score = df_score * w_rep
        score = pd.DataFrame(score.sum(axis=1))
        score.columns = ['score']
    
        return w, score
    
    return w, None
    
 
if __name__ == '__main__':
#    df = pd.read_csv('../test/EntValWeight_test.csv').dropna(how='any')
    
#    df = pd.read_csv('../test/GDP2015.csv', encoding='gbk').set_index('地区')
#    indexs = ['GDP总量增速', '人口总量', '人均GDP增速', '地方财政收入总额',
#              '固定资产投资', '社会消费品零售总额增速', '进出口总额',
#              '城镇居民人均可支配收入', '农村居民人均可支配收入']
#    df = df.reindex(columns=indexs).dropna(how='any')
    
    print('熵值法有个明显缺点（原因是熵值法缺少了指标之间的横向比较？）：')
    print('比如下面这个例子中，从专家打分看，经济效应的重要性应高于社会效应，' + \
          '但是两者权重却相等')
    data = pd.read_csv('../test/指标重要性专家评分表1.csv', encoding='gbk')
    indexs = ['经济效益', '社会效益', '生态效益']
    df = data.set_index('专家').reindex(columns=indexs).dropna(how='any')
    print(df)

    neg_cols, score_type = [], None
    w, score = EntValWeight(df, neg_cols=neg_cols, score_type=score_type)
    print(w)
    print(score)
        