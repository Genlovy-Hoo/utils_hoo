# -*- coding: utf-8 -*-

from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from .utils_general import simple_logger
# from utils_hoo.utils_general import simple_logger

#%%
class MultiThread(Thread):
    '''
    多线程子任务类
    '''
    
    def __init__(self, func, args, logger=None):
        '''
        Args:
            func: 需要多线程运行的目标函数
            args: 目标函数func输入变量列表
        '''        
        super(MultiThread, self).__init__()
        self.func = func
        self.args = args
        if logger is None:
            logger = simple_logger()
        self.logger = logger
        
    def run(self):
        '''执行目标函数func，获取返回结果'''
        self.result = self.func(*self.args)
        
    def get_result(self):
        '''获取执行结果'''
        try:
            return self.result
        except:
            self.logger.error('error found, return None.', exc_info=True)
            return None
        
            
def multi_thread_threading(func, args_list, logger=None):
    '''
    多线程，同一个函数执行多次
    
    Parameters
    ----------
    func: 需要多线程运行的目标函数
    args_list: list，每个元素都是目标函数func的参数列表
    logger: logging库的日志记录器
        
    Returns
    -------
    results: list，每个元素对应func以args_list的元素为输入的返回结果
    '''
    
    tasks = []
    for args in args_list:
        task = MultiThread(func, args, logger=logger)
        tasks.append(task)
        task.start()
    
    results = []
    for task in tasks:
        task.join()
        results.append(task.get_result())
        
    return results
    
#%%
def multi_thread_concurrent(func, args_list, multi_line=None, keep_order=True):
    '''
    多线程，同一个函数执行多次
    
    Parameters
    ----------
    func: 需要多线程运行的目标函数
    args_list: list，每个元素都是目标函数func的参数列表
    multi_line: 最大线程数，默认等于len(args_list)
    keep_order: 是否保持输入args_list与输出results参数顺序一致性，默认是
        
    Returns
    -------
    results: list，每个元素对应func以args_list的元素为输入的返回结果
    '''
    
    if multi_line is None:
        multi_line = len(args_list)
        
    # submit方法不能保证results的值顺序与args_list一一对应
    if not keep_order:
        with ThreadPoolExecutor(max_workers=multi_line) as executor:
            futures = [executor.submit(func, *args) for args in args_list]
                       
        results = []
        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    # 使用map可保证results的值顺序与args_list一一对应
    if keep_order:
        def func_new(args):
            return func(*args)
            
        with ThreadPoolExecutor(max_workers=multi_line) as executor:
            results = executor.map(func_new, args_list)
            results = list(results)
        
    return results
    
#%%
if __name__ == '__main__':    
    import time
    
    
    def func(idx, sleep_tm):
        print('task id:', idx)
        time.sleep(sleep_tm)
        print('task id: {}; slept: {}s.'.format(idx, sleep_tm))
        return [idx, sleep_tm]
    
    args_list = [[1, 2], [3, 4], [4, 5], [2, 3]]


    print('multi-thread, threading..............................')
    strt_tm = time.time()
    results_threading = multi_thread_threading(func, args_list)   
    print('used time: {tm}s.'.format(tm=round(time.time() - strt_tm,6)))
    
    
    print('multi-thread, concurrent.............................')
    strt_tm = time.time()
    results_concurrent_Order = multi_thread_concurrent(func, args_list,
                                                     keep_order=True)
    results_concurrent_noOrder = multi_thread_concurrent(func, args_list,
                                                       keep_order=False)    
    print('used time: {tm}s.'.format(tm=round(time.time() - strt_tm,6)))
    