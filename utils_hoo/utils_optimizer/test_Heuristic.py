# -*- coding: utf-8 -*-

import time
import pandas as pd
from utils_hoo.utils_optimizer.GA import GA
from utils_hoo.utils_optimizer.CS import CS
from utils_hoo.utils_optimizer.PSO import PSO
from utils_hoo.utils_optimizer.GWO import GWO
from utils_hoo.utils_optimizer.WOA import WOA
from utils_hoo.utils_optimizer.HHO import HHO
from utils_hoo.utils_general import simple_logger
from utils_hoo.utils_plot.plot_Common import plot_Series
from utils_hoo.utils_optimizer.test_funcs import TestFuncs
from utils_hoo.utils_logging.logger_general import get_logger
from utils_hoo.utils_logging.logger_utils import close_log_file
from utils_hoo.utils_optimizer.utils_Heuristic import FuncOpterInfo


if __name__ == '__main__':
    strt_tm = time.time()
    
    # 目标函数和参数
    objf = TestFuncs.F7
    parms_func = {'func_name': objf.__name__,
                  'x_lb': -30, 'x_ub': 30, 'dim': 100, 'kwargs': {}}
    
    # 统一参数
    PopSize = 30
    Niter = 1000
    
    # logger
    # logger = simple_logger()
    logger = get_logger('./test/Heuristic-test.txt', screen_show=True)
    # parms_log = {'logger': logger, 'nshow': 10}
    parms_log = {'logger': logger, 'nshow': Niter}
    
    fvals = pd.DataFrame()
    
    # GA
    parms_ga = {'opter_name': 'GA',
                'PopSize': PopSize, 'Niter': Niter,
                'Pcrs': 0.7, 'Pmut': 0.1, 'Ntop': 2}
    
    ga_parms = FuncOpterInfo(parms_func, parms_ga, parms_log)
    ga_parms = GA(objf, ga_parms)
    fvals['GA'] = ga_parms.convergence_curve
    
    # PSO
    parms_pso = {'opter_name': 'PSO',
                 'PopSize': PopSize, 'Niter': Niter,
                 'v_maxs': 5, 'w_max': 0.9, 'w_min': 0.2, 'w_fix': False,
                 'c1': 2, 'c2': 2}
    
    pso_parms = FuncOpterInfo(parms_func, parms_pso, parms_log)
    pso_parms = PSO(objf, pso_parms)
    fvals['PSO'] = pso_parms.convergence_curve
    
    # CS
    parms_cs = {'opter_name': 'CS',
                'PopSize': PopSize, 'Niter': Niter,
                'pa': 0.25, 'beta': 1.5, 'alpha': 0.01}
    
    cs_parms = FuncOpterInfo(parms_func, parms_cs, parms_log)
    cs_parms = CS(objf, cs_parms)
    fvals['CS'] = cs_parms.convergence_curve
    
    # GWO
    parms_gwo = {'opter_name': 'GWO',
                 'PopSize': PopSize, 'Niter': Niter}
    
    gwo_parms = FuncOpterInfo(parms_func, parms_gwo, parms_log)
    gwo_parms = GWO(objf, gwo_parms)
    fvals['GWO'] = gwo_parms.convergence_curve
    
    # WOA
    parms_woa = {'opter_name': 'WOA',
                 'PopSize': PopSize, 'Niter': Niter}
    
    woa_parms = FuncOpterInfo(parms_func, parms_woa, parms_log)
    woa_parms = WOA(objf, woa_parms)
    fvals['WOA'] = woa_parms.convergence_curve
    
    # HHO
    parms_hho = {'opter_name': 'HHO',
                 'PopSize': PopSize, 'Niter': Niter,
                 'beta': 1.5, 'alpha': 0.01}
    
    hho_parms = FuncOpterInfo(parms_func, parms_hho, parms_log)
    hho_parms = HHO(objf, hho_parms)
    fvals['HHO'] = hho_parms.convergence_curve
    
    
    # 参数汇总
    Results = pd.DataFrame({'ga': ga_parms.best_x,
                            'pso': pso_parms.best_x,
                            'cs': cs_parms.best_x,
                            'gwo': gwo_parms.best_x,
                            'woa': woa_parms.best_x,
                            'hho': hho_parms.best_x})
    
    
    # 作图比较
    plot_Series(fvals.iloc[150:, :],
                {'GA': '-', 'PSO': '-', 'CS': '-', 'GWO': '-', 'WOA': '-',
                 'HHO': '-'},
                figsize=(10, 6))
    
    
    close_log_file(logger)
    
    
    print(f'used time: {round(time.time()-strt_tm, 6)}s.')
