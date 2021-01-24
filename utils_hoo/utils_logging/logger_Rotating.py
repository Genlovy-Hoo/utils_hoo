# -*- coding: utf-8 -*-


from .logger_utils import remove_handlers
from .logger_utils import close_log_file
import logging
from logging.handlers import RotatingFileHandler
import os
        
        
def get_logger(fpath=None, fmode='w', maxK=1, bkupCount=3, screen_show=True):
    '''
    滚动日志记录（按文件大小），将日志信息滚动保存在文件或在屏幕中打印
    
    Args:
        fapth: 日志文件路径，默认为None即不保存日志文件
        fmode: 'w'或'a'，取'w'时会覆盖原有日志文件（若存在的话），取'a'则追加记录
        maxK: 单个日志文件大小限制，单位为kb
        bkupCount: 最多备份文件个数
        screen_show: 是否在控制台打印日志信息，默认打印
        注: fpath和screen_show必须有至少一个为真
        
    Return:
        logger: logging的日志记录器
    '''
    
    if fpath is None and not screen_show:
        raise ValueError('必须设置日志文件路径或在屏幕打印日志！')
    
    # 准备日志记录器logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    # 预先删除logger中已存在的handlers
    logger = remove_handlers(logger)
    
    # 日志格式
    formatter = logging.Formatter(
    '''%(asctime)s -%(filename)s[line: %(lineno)d] -%(levelname)s:
    --%(message)s''')
    
    if fpath is not None:
        if fmode == 'w' and os.path.exists(fpath):
            # 先删除原有日志文件
            fdir = os.path.dirname(os.path.realpath(fpath))
            fname = os.path.basename(fpath)
            fpaths = [x for x in os.listdir(fdir) if fname in x]
            for f in fpaths:
                os.remove(os.path.join(fdir, f))
        # 日志文件保存，FileHandler
        file_logger = RotatingFileHandler(fpath, mode=fmode,
                                          maxBytes=maxK*1024,
                                          backupCount=bkupCount)
        file_logger.setLevel(logging.DEBUG)
        file_logger.setFormatter(formatter)
        logger.addHandler(file_logger)
        
    if screen_show:
        # 控制台打印，StreamHandler
        console_logger = logging.StreamHandler()
        console_logger.setLevel(logging.DEBUG)
        console_logger.setFormatter(formatter)
        logger.addHandler(console_logger)
        
    return logger


if __name__ == '__main__':
    log_path = './test_log/log_test2.log'
    logger = get_logger(fpath=log_path, fmode='w', maxK=1, bkupCount=3,
                        screen_show=True)
    
    logger.info('Log start here ********************************************')
    logger.debug('Do something.')
    logger.warning('Something maybe fail.')
    logger.error('Some error find here.')
    logger.critical('Program crashed.')
    logger.info('Finish')
    
    logger.info('Log start here ---------------------------------------------')
    logger.debug('Do something.')
    logger.warning('Something maybe fail.')
    logger.error('Some error find here.')
    logger.critical('Program crashed.')
    logger.info('Finish')
    
    logger.info('Log start here #############################################')
    logger.debug('Do something.')
    logger.warning('Something maybe fail.')
    logger.error('Some error find here.')
    logger.critical('Program crashed.')
    logger.info('Finish')
    
    close_log_file(logger)
    
    

