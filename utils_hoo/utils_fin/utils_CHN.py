# -*- coding: utf-8 -*-

import os
import pandas as pd
from datetime import datetime
from utils_hoo import load_csv
from chinese_calendar import is_workday
from utils_hoo.utils_general import isnull
from utils_hoo import utils_datetime as utl_dttm


def get_code_ext(code):
    '''
    返回带交易所后缀的股票代码格式，如输入`300033`，返回`300033.SZ`
    code目前可支持[A股、B股、50ETF期权、300ETF期权]，根据需要更新
    如不能确定后缀，则直接返回code原始值
    
    http://www.sse.com.cn/lawandrules/guide/jyznlc/jyzn/c/c_20191206_4960455.shtml
    '''
    
    code = str(code)
    
    # 上交所A股以'600'、'601'、'603'、'688'（科创板）开头，B股以'900'开头，共6位
    if len(code) == 6 and code[0:3] in ['600', '601', '603', '688', '900']:
        return code + '.SH'
    
    # 上交所50ETF期权和300ETF期权代码以'100'开头，共8位
    if len(code) == 8 and code[0:3] == '100':
        return code + '.SH'
    
    # 深交所A股以'000'（主板）、'002'（中小板）, '300'（创业板）开头，共6位
    # 深交所B股以'200'开头，共6位
    if len(code) == 6 and code[0:3] in ['000', '002', '300', '200']:
        return code + '.SZ'
    
    # 深交所300ETF期权代码以'900'开头，共8位
    if len(code) == 8 and code[0:3] == '900':
        return code + '.SZ'
    
    return code


def get_trade_fee_Astock(code, BorS, vol, price,
                         fee_least=5, fee_pct=2.5/10000):
    '''
    普通A股股票普通交易费用计算
    '''
    if str(code)[0] == '6':
        return trade_fee_Astock('SH', BorS, vol, price, fee_least, fee_pct)
    else:
        return trade_fee_Astock('SZ', BorS, vol, price, fee_least, fee_pct)


def trade_fee_Astock(mkt, BorS, vol, price, fee_least=5, fee_pct=2.5/10000):
    '''
    普通A股股票普通交易费用计算
    
    Parameters
    ----------
    mkt: 'SH'('sh', 'SSE')或'SZ'('sz', 'SZSE')，分别代表上海和深圳市场
    BorS: 'B'('b', 'buy')或'S'('s', 'sell', 'sel')，分别标注买入或卖出
    vol: 量（股）
    price: 价格（元）
    fee_least和fee_pct设置券商手续费最低值和比例
    
    Returns
    -------
    fee_mkt + fee_sec: 交易成本综合
    
    收费标准原因沪深交易所官网，若有更新须更改：
    http://www.sse.com.cn/services/tradingservice/charge/ssecharge/（2020年4月）
    http://www.szse.cn/marketServices/deal/payFees/index.html（2020年2月）
    '''
    
    if mkt in ['SH', 'sh', 'SSE']:
        if BorS in ['B', 'b', 'buy']:
            tax_pct = 0.0 / 1000 # 印花税
            sup_pct = 0.2 / 10000 # 证券交易监管费
            hand_pct = 0.487 / 10000 # 经手（过户）费
        elif BorS in ['S', 's', 'sell', 'sel']:
            tax_pct = 1.0 / 1000
            sup_pct = 0.2 / 10000
            hand_pct = 0.487 / 10000
            
        net_cash = vol * price # 交易额
        fee_mkt = net_cash * (tax_pct + sup_pct + hand_pct) # 交易所收费
        fee_sec = max(fee_least, net_cash * fee_pct) # 券商收费
        
    if mkt in ['SZ', 'sz', 'SZSE']:
        if BorS in ['B', 'b', 'buy']:
            tax_pct = 0.0 / 1000 # 印花税
            sup_pct = 0.2 / 10000 # 证券交易监管费
            hand_pct = 0.487 / 10000 # 经手（过户）费
        elif BorS in ['S', 's', 'sell', 'sel']:
            tax_pct = 1.0 / 1000
            sup_pct = 0.2 / 10000
            hand_pct = 0.487 / 10000
            
        net_cash = vol * price # 交易额
        fee_mkt = net_cash * (tax_pct + sup_pct + hand_pct) # 交易所收费
        fee_sec = max(fee_least, net_cash * fee_pct) # 券商收费
        
    return fee_mkt + fee_sec


def IsTradeDay_ChnCal(date):
    '''
    利用chinese_calendar库判断date（str格式）是否为交易日
    注：若chinese_calendar库统计的周内工作日与交易日有差异或没更新，可能导致结果不准确
    '''
    date = utl_dttm.date_reformat(date, '')
    date_dt = datetime.strptime(date, '%Y%m%d')
    return is_workday(date_dt) and date_dt.weekday() not in [5, 6]


def get_recent_trade_date_ChnCal(date):
    '''
    若date为交易日，则返回，否则返回下一个交易日
    注：若chinese_calendar库统计的周内工作日与交易日有差异或没更新，可能导致结果不准确
    '''
    while not IsTradeDay_ChnCal(date):
        date = utl_dttm.date_add_Nday(date, 1)
    return date
    
    
def get_next_nth_trade_date_ChnCal(date, n=1):
    '''
    给定日期date，返回其后第n个交易日日期，n可为负数（返回结果在date之前）
    注：若chinese_calendar库统计的周内工作日与交易日有差异或没更新，可能导致结果不准确
    '''
    n_add = -1 if n < 0 else 1
    n = abs(n)
    tmp = 0
    while tmp < n:
        date = utl_dttm.date_add_Nday(date, n_add)
        if IsTradeDay_ChnCal(date):
            tmp += 1
    return date


def get_next_nth_trade_date(trade_dates_df_path, date, n=1):
    '''
    给定日期date，返回其后第n个交易日日期，n可为负数（返回结果在date之前）
    trade_dates_df_path可以为历史交易日期数据存档路径，也可以为pd.DataFrame
    注：默认trade_dates_df_path数据格式为tushare格式（GetData1_tushare.py）：
        exchange,date,is_open
        SSE,2020-09-02,1
        SSE,2020-09-03,1
        SSE,2020-09-04,1
        SSE,2020-09-05,0
    '''
    _, joiner = utl_dttm.get_format(date)
    date = utl_dttm.date_reformat(date, '-')
    if isinstance(trade_dates_df_path, str) \
                                    and os.path.isfile(trade_dates_df_path):
        dates = load_csv(trade_dates_df_path)
    else:
        dates = trade_dates_df_path.copy()
    dates.sort_values('date', ascending=True, inplace=True)
    dates.drop_duplicates(subset=['date'], keep='last', inplace=True)
    dates['tmp'] = dates[['date', 'is_open']].apply(lambda x:
                1 if x['is_open'] == 1 or x['date'] == date else 0, axis=1)
    dates = list(dates[dates['tmp'] == 1]['date'].unique())
    dates.sort()
    idx = dates.index(date)
    if -1 < idx+n < len(dates):
        return utl_dttm.date_reformat(dates[idx+n], joiner)
    else:
        return None


def get_trade_dates_ChnCal(start_date, end_date, joiner='-'):
    '''
    利用chinese_calendar库获取指定起止日期内的交易日期（周内的工作日）
    注：若chinese_calendar库统计的周内工作日与交易日有差异或没更新，可能导致结果不准确
    '''
    _, joiner = utl_dttm.get_format(start_date)
    dates = pd.date_range(start_date, end_date)
    dates = [x.strftime(joiner.join(['%Y', '%m', '%d'])) for x in dates if \
                        is_workday(x) and x.weekday() not in [5, 6]]
    dates = [utl_dttm.date_reformat(x, joiner) for x in dates]
    return dates
    
    
def get_trade_dates(trade_dates_df_path, start_date, end_date):
    '''
    获取起止日期之间(从start_date到end_date)的交易日期，返回列表
    trade_dates_df_path可以为历史交易日期数据存档路径，也可以为pd.DataFrame
    注：默认trade_dates_df_path数据格式为tushare格式（GetData1_tushare.py）：
        exchange,date,is_open
        SSE,2020-09-02,1
        SSE,2020-09-03,1
        SSE,2020-09-04,1
        SSE,2020-09-05,0
    '''
    _, joiner = utl_dttm.get_format(start_date)
    start_date = utl_dttm.date_reformat(start_date, '-')
    end_date = utl_dttm.date_reformat(end_date, '-')
    if isinstance(trade_dates_df_path, str) \
                                    and os.path.isfile(trade_dates_df_path):
        data = load_csv(trade_dates_df_path)
    else:
        data = trade_dates_df_path.copy()
    data = data[data['date'] >= start_date]
    data = data[data['date'] <= end_date]
    dates = data[data['is_open'] == 1]['date']
    dates = [utl_dttm.date_reformat(x, joiner) for x in list(dates)]
    return dates


def get_num_trade_dates(start_date, end_date, trade_dates_df_path=None):
    '''给定起止时间获取可交易天数'''
    if not isnull(trade_dates_df_path):
        trade_dates = get_trade_dates(trade_dates_df_path, start_date,
                                      end_date)
    else:
        trade_dates = get_trade_dates_ChnCal(start_date, end_date)
    return len(trade_dates)
