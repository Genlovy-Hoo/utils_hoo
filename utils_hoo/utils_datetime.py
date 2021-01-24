# -*- coding: utf-8 -*-

import re
import datetime
from chinese_calendar import is_workday


def get_format(date, joiners=[' ', '-', '/', '*', '#', '@', '.', '_']):
    '''
    判断日期格式位，返回日期位数和连接符
    
    Args:
        date: 表示日期的字符串
        joiners: 支持的格式连接符，不支持的连接符可能报错或返回'未知日期格式'
                 注：若date无格式连接符，则只判断8位的格式，如'20200220'
    
    Return:
        (日期位数（可能为date字符串长度或'未知日期格式'）, 格式连接符（或None）)
    '''
    
    for joiner in joiners:
        reg = re.compile(r'\d{4}['+joiner+']\d{2}['+joiner+']\d{2}')
        if len(reg.findall(date)) > 0:
            return len(date), joiner
        
    # 这里只要8位都是数字就认为date是形如'20200226'格式的日期
    if len(date) == 8:
        tmp = [str(x) for x in range(0, 10)]
        tmp = [x in tmp for x in date]
        if all(tmp):
            return 8, ''
        
    return '未知日期格式', None


def get_YearMonth(date):
    '''提取date中的年月。eg. `20200305`—>`202003`, `2020-03-05`—>`2020-03`'''
    n, joiner = get_format(date)
    if n == 8 and joiner == '':
        return date[0:-2]
    if joiner is not None:
        return joiner.join(date.split(joiner)[0:2])
    raise ValueError('请检查日期格式：只接受get_format函数识别的日期格式！')


def today_date(joiner='-', forbid_joiners=['%']):
    '''
    获取今日日期，格式由连接符joiner确定，forbid_joiners指定禁用的连接符
    '''
    if joiner in forbid_joiners:
        raise ValueError('非法连接符：{}！'.format(joiner))
    return datetime.datetime.now().strftime(joiner.join(['%Y', '%m', '%d']))


def date_reformat(date, joiner='-', forbid_joiners=['%']):
    '''指定连接符为joiner，重新格式化date，forbid_joiners指定禁用的连接符'''
    if joiner in forbid_joiners:
        raise ValueError('非法连接符：{}！'.format(joiner))    
    n, joiner_ori = get_format(date)
    if joiner_ori is not None:
        formater_ori = joiner_ori.join(['%Y', '%m', '%d'])
        date = datetime.datetime.strptime(date, formater_ori)
        formater = joiner.join(['%Y', '%m', '%d'])
        return date.strftime(formater)
    raise ValueError('请检查日期格式：只接受get_format函数识别的日期格式！')
    
    
def date8_to_date10(date, joiner='-'):
    '''8位日期转换为10位日期，连接符为joiner'''
    n, _ = get_format(date)
    if n != 8:
        raise ValueError('请检查日期格式，接受8位且get_format函数识别的日期格式！')
    return joiner.join([date[0:4], date[4:6], date[6:]])


def date10_to_date8(date):
    '''10位日期转换为8位日期'''
    n, joiner = get_format(date)
    if n != 10 or joiner is None:
        raise ValueError('请检查日期格式，接受10位且get_format函数识别的日期格式！')
    return ''.join(date.split(joiner))
    
    
def date_add_Nday(date, N=1):
    '''
    在给定日期date上加上N天（减去时N写成负数即可）
    日期输入输出符合get_format函数支持的格式
    '''
    n, joiner = get_format(date)
    if joiner is not None:
        formater = joiner.join(['%Y', '%m', '%d'])
        date = datetime.datetime.strptime(date, formater)
        date_delta = datetime.timedelta(days=N)
        date_new = date + date_delta
        return date_new.strftime(formater)
    raise ValueError('请检查日期格式：只接受get_format函数识别的日期格式！')
    
    
def diff_days_date(date1, date2):
    '''
    计算两个日期间相隔天数，若date1大于date2，则输出为正，否则为负
    日期输入输出符合get_format函数支持的格式
    '''
    n1, joiner1 = get_format(date1)
    n2, joiner2 = get_format(date2)    
    if (n1, joiner1) != (n2, joiner2):
        raise ValueError('两个日期格式不相同，请检查！')
    if joiner1 is not None:
        formater = joiner1.join(['%Y', '%m', '%d'])
        date1 = datetime.datetime.strptime(date1, formater)
        date2 = datetime.datetime.strptime(date2, formater) 
        return (date1-date2).days
    raise ValueError('请检查日期格式：只接受get_format函数识别的日期格式！')
    
    
def IsWorkDay_ChnCal(date):
    '''利用chinese_calendar库判断date（str格式）是否为工作日'''
    date = date_reformat(date, '')
    date_dt = datetime.datetime.strptime(date, '%Y%m%d')
    return is_workday(date_dt)
    
    
def diff_workdays_date(date1, date2):
    '''
    计算两个日期间相隔的工作日天数，若date1大于date2，则输出为正，否则为负
    日期输入输出符合get_format函数支持的格式
    '''
    raise NotImplementedError
