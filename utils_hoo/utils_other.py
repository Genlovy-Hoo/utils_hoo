# -*- coding: utf-8 -*-

from .utils_io import load_csv

def load_csv_ColMaxMin(csv_path, col='date', return_data=True, dropna=False,
                       **kwargs):
    '''
    获取csv_path历史数据col列（csv文件必须有col列）的最大值和最小值
    当return_data为True时返回最大值、最小值和df数据，为False时不返回数据（None）
    dropna设置在判断最大最小值之前是否删除col列的无效值
    **kwargs为load_csv可接受的参数
    '''
    data = load_csv(csv_path, **kwargs)
    if dropna:
        data.dropna(how='any', inplace=True)
    data.sort_values(col, ascending=True, inplace=True)    
    col_Max, col_Min = data[col].iloc[-1], data[col].iloc[0]
    if return_data:
        return col_Max, col_Min, data
    else:
        return col_Max, col_Min, None