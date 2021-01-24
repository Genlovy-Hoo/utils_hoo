# -*- coding: utf-8 -*-


from .logger_utils import remove_handlers
from .logger_utils import close_log_file
import logging
import os
        
        
def get_logger(fpath=None, fmode='w', screen_show=True):
    '''
    常规日志记录，将日志信息保存在文件或在屏幕中打印
    
    Args:
        fapth: 日志文件路径，默认为None即不保存日志文件
        fmode: 'w'或'a'，取'w'时会覆盖原有日志文件（若存在的话），取'a'则追加记录
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
            os.remove(fpath)
        # 日志文件保存，FileHandler
        file_logger = logging.FileHandler(fpath, mode=fmode)
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
    log_path = './test_log/log_test1.log'
    logger = get_logger(fpath=log_path, fmode='w', screen_show=True)
    
    logger.info('Log start here ********************************************')
    logger.debug('Do something.')
    logger.warning('Something maybe fail.')
    logger.error('Some error find here.')
    logger.critical('Program crashed.')
    logger.info('Finish')
    
    close_log_file(logger)
    
    

