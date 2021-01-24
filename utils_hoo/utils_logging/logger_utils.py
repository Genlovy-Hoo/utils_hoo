# -*- coding: utf-8 -*-


import logging


def close_log_file(logger):
    '''关闭日志记录器logger中的文件流'''
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            h.close()
    return logger
            
        
def remove_handlers(logger):
    '''
    关闭并移除logger中已存在的handlers
    注：这里貌似必须先把FileHandler close并remove之后，
       再remove其它handler才能完全remove所有handlers，原因待查（可能由于FileHandler
       是StreamHandler的子类的缘故）
    '''
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            h.close()
            logger.removeHandler(h)
    for h in logger.handlers:
            logger.removeHandler(h)
    return logger
