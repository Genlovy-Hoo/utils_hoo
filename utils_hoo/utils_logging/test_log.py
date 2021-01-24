# -*- coding: utf-8 -*-

from utils_hoo.utils_logging import logger_utils
from utils_hoo.utils_logging import logger_general
from utils_hoo.utils_logging import logger_Rotating
from utils_hoo.utils_logging import logger_TimedRotating
from test_func import test_func
import time


if __name__ == '__main__':
    #%%
    logger = logger_general.get_logger(fpath='./test_log/log_test1.log',
                                       fmode='w',
                                       screen_show=True)
    
    logger.info('Log start here *********************************************')
    test_func(3, 5, 'this is a warning.', logger)
    logger.info('测试结束.')
    
    logger_utils.close_log_file(logger)
    logger_utils.remove_handlers(logger)
    
    #%%
    logger = logger_Rotating.get_logger(fpath='./test_log/log_test2.log',
                                        fmode='w', maxK=1, bkupCount=3,
                                        screen_show=True) 
    logger.info('Log start here *********************************************')
    test_func(3, 1, 'this is a warning.', logger)
    logger.info('测试结束.')
    
    logger_utils.close_log_file(logger)
    logger_utils.remove_handlers(logger)
    
    #%%
    log_path = './test_log/log_test3.log'
    logger = logger_TimedRotating.get_logger(fpath=log_path, when='S',
                                             interval=3, bkupCount=3,
                                             screen_show=True) 
    
    count = 0
    while count < 2:
        logger.info('{}th log start here ********************'.format(count+1))
        test_func(3, 2, 'this is a warning.', logger)
        logger.info('测试结束.')
        
        time.sleep(2)
        count += 1
    
    logger_utils.close_log_file(logger)
    logger_utils.close_log_file(logger)
    