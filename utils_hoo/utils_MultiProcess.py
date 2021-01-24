# -*- coding: utf-8 -*-

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import time
import platform


def multi_process_concurrent(func, args_list, keep_order=True,
                             multi_line=None):
    '''    
    多进程，同一个函数执行多次
    
    Parameters
    ----------
    func: 需要多进程运行的目标函数
    args_list: list，每个元素都是目标函数func的参数列表
    keep_order: 是否保持输入args_list与输出results参数顺序一致性
                若keep_order为True，则func的格式应转化为：
                def func(args):
                    return f(*args)
    multi_line: 最大线程数，默认等于len(args_list)
        
    Returns
    -------
    results: list，每个元素对应func以args_list的元素为输入的返回结果
        
    注：该函数通过import导入在Windows下会出错
    '''
    
    if multi_line is None:
        multi_line = len(args_list)
    
    # submit方法不能保证results的值顺序与args_list一一对应
    if not keep_order:
        with ProcessPoolExecutor(max_workers=multi_line) as executor:
            futures = [executor.submit(func, *args) for args in args_list]       
                       
        results = []
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            
    # 使用map可保证results的值顺序与args_list一一对应
    if keep_order:
            
        with ProcessPoolExecutor(max_workers=multi_line) as executor:
            results = executor.map(func, args_list)
            results = list(results)
        
    return results
    
#%%   
def func_test_win(idx, sleep_tm):
    '''注：若此目标函数定义在__name__ == '__main__'之后，则在windows下会报错'''
    print('task id:', idx)
    time.sleep(sleep_tm)
    print('task id: {}; slept: {}s.'.format(idx, sleep_tm))
    return [idx, sleep_tm]


def func_test_win_new(args):
    return func_test_win(*args)
    
#%%
if __name__ == '__main__': 
    
    def func(idx, sleep_tm):
        '''注：若此目标函数定义在__name__ == '__main__'之后，则在windows下会报错'''
        print('task id:', idx)
        time.sleep(sleep_tm)
        print('task id: {}; slept: {}s.'.format(idx, sleep_tm))
        return [idx, sleep_tm]
    
    def func_new(args):
        return func(*args)
    
    args_list = [[1, 2], [3, 4], [4, 5], [2, 3]]
    
    
    print('multi-process, concurrent............................')
    strt_tm = time.time()
    if platform.system() == 'Windows':
        results_Order = multi_process_concurrent(func_test_win_new, args_list,
                                                 keep_order=True)   
        results_noOrder = multi_process_concurrent(func_test_win, args_list,
                                                   keep_order=False)
    else:
        results_Order = multi_process_concurrent(func_new, args_list,
                                                 keep_order=True)   
        results_noOrder = multi_process_concurrent(func, args_list,
                                                   keep_order=False)
    print('total used time: {tm}s.'.format(tm=round(time.time() - strt_tm,6)))  
    