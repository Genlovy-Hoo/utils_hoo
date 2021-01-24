# -*- coding: utf-8 -*-

import os
import pickle
import pandas as pd
import shutil
import zipfile
import subprocess
from .utils_general import simple_logger, isnull
# from utils_hoo.utils_general import simple_logger, isnull

def copy_file():
	'''复制文件'''
	raise NotImplementedError
	
	
def copy_dir():
    '''复制文件夹'''
    raise NotImplementedError
    
    
def load_json(file):
    '''读取json格式文件file'''
    raise NotImplementedError
    
    
def write_json(data, file):
    '''把data写入json格式文件file'''
    raise NotImplementedError


def extract_7z(zip_path, save_dir):
    '''
    7z命令解压文件
    '''
    raise NotImplementedError


def zip_fpath_7z(fpath, zip_path=None, mode='zip', pwd=None,
                 keep_zip_new=True):
    '''
    7z命令压缩单个文件（夹）
    
    fpath: 待压缩文件(夹)路径
    zip_path: 压缩文件保存路径，若为None，则为fpath路径加后缀
    mode: 压缩文件后缀，可选['7z', 'zip']
    pwd: 密码字符串
    keep_zip_new: 为True时若zip_path已经存在, 则会先删除已经存在的再重新创建新压缩文件
    '''
    
    fpath = os.path.abspath(fpath) # 绝对路径
    
    if isnull(zip_path):
        zip_path = fpath + '.zip'
    else:
        zip_path = os.path.abspath(zip_path)
        
    if os.path.exists(zip_path) and keep_zip_new:
        os.remove(zip_path)
        
    md_str = ' -t' + mode
    
    if isnull(pwd):
        pwd = ''
    else:
        pwd = ' -p' + str(pwd)
        
    cmd_str = '7z a ' + zip_path + ' ' + fpath + md_str + pwd
    
    # os.system(cmd_str) # windows下会闪现cmd界面
    subprocess.call(cmd_str, shell=True)
    
    
def zip_fpaths_7z(zip_path, fpaths, mode='zip', pwd=None, keep_zip_new=True):
    '''
    7z命令压缩多个文件（夹）列表files到压缩文件zip_path
    注: files中的单个文件(不是文件夹)在压缩包中不会保留原来的完整路径, 
        所有单个文件都会在压缩包根目录下
    
    zip_path: 压缩文件保存路径
    fpaths: 压缩文件路径列表，fpaths太长的时候会出错
    mode: 压缩文件后缀，可选['7z', 'zip']
    pwd: 密码字符串
    keep_zip_new: 为True时若zip_path已经存在, 则会先删除已经存在的再重新创建新压缩文件
    '''
    
    md_str = ' -t' + mode
    
    if isnull(pwd):
        pwd = ''
    else:
        pwd = ' -p' + str(pwd)
        
    if os.path.exists(zip_path) and keep_zip_new:
        os.remove(zip_path)
        
    fpaths_str = ' '.join([os.path.abspath(x) for x in fpaths])
    
    cmd_str = '7z a ' + zip_path + ' ' + fpaths_str + md_str + pwd

    # os.system(cmd_str) # windows下会闪现cmd界面
    subprocess.call(cmd_str, shell=True)
    
    
def zip_fpath(fpath, zip_path=None):
    '''
    zipfile压缩单个文件（夹）
    
    fpath: 待压缩文件夹路径(应为相对路径)
    zip_path: 压缩文件保存路径，若为None，则为fpath路径加后缀
    '''
    
    if isnull(zip_path):
        if os.path.isdir(fpath) and fpath[-1] == '/':
            zip_path = fpath[:-1] + '.zip'
        else:
            zip_path = fpath + '.zip'
    
    if os.path.isfile(fpath): # fpath为文件
        zip_files(zip_path, [fpath])
    elif os.path.isdir(fpath): # fpath为文件夹
        fpaths = get_all_files(fpath)
        zip_files(zip_path, fpaths)
        
        
def zip_fpaths(zip_path, fpaths):
    '''
    压缩路径列表fpaths(可为文件也可为文件夹, 应为相对路径)到zip_path
    zip_path：zip压缩包保存路径
    fpaths：需要压缩的路径列表(应为相对路径, 可为文件也可为文件夹)
    '''
    all_paths = []
    for fpath in fpaths:
        if os.path.isfile(fpath):
            all_paths.append(fpath)
        elif os.path.isdir(fpath):
            all_paths += get_all_files(fpath)
    zip_files(zip_path, all_paths)
    

def zip_files(zip_path, fpaths, keep_ori_path=True):
    '''
    使用zipfile打包为.zip文件
    zip_path：zip压缩包保存路径
    fpaths：需要压缩的文件(不是能文件夹)路径列表(应为相对路径)
    keep_ori_path: 若为True, 则压缩文件会保留fpaths中文件的原始路径
                   若为False, 则fpaths中所有文件在压缩文件中都在统一根目录下
    '''
    f = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)
    if keep_ori_path:
        for fpath in fpaths:
            f.write(fpath)
    else:
        for fpath in fpaths:
            file = os.path.basename(fpath)
            f.write(fpath, file)
    f.close()
    
    
def get_all_files(dir_path):
    '''获取dir_path文件夹及其子文件夹中所有的文件路径'''
    fpaths = []
    for root, dirs, files in os.walk(dir_path):
        for fname in files:
            fpaths.append(os.path.join(root, fname))
    return fpaths


def del_dir(dir_path):
    '''删除文件夹及其所有内容'''
    shutil.rmtree(dir_path)


def pickleFile(data, file):
    '''以二进制格式保存数据data到文件file'''
    with open(file, 'wb') as dbFile:
        pickle.dump(data, dbFile)
        
        
def unpickeFile(file):
    '''读取二进制格式文件file'''
    with open(file, 'rb') as dbFile:
        return pickle.load(dbFile)
    
    
def write_txt(lines, file, mode='w'):
    '''
    将lines写入txt文件，文件路径为file
    lines为列表，每个元素为一行文本内容，末尾不包括换行符
    mode为写入模式，如'w'或'a'
    '''
    lines = [line + '\n' for line in lines]
    f = open(file, mode=mode)
    f.writelines(lines)
    f.close()
    

def load_text(fpath, sep=',', del_first_line=False, del_first_col=False,
              to_pd=True, keep_header=True, encoding=None, logger=None):
    '''
    读取文本文件数据，要求文件每行一个样本
    
    Args:
        fpath: 文本文件路径
        sep: 字段分隔符，默认`,`
        del_first_line: 是否删除首行，默认不删除
        del_first_col=False: 是否删除首列，默认不删除
        to_pd: 是否输出为pandas.DataFrame，默认是
        keep_header: 输出为pandas.DataFrame时是否以首行作为列名，默认是
        encoding: 指定编码方式，默认不指定，不指定时会尝试以uft-8和gbk编码读取
        logger: 日志记录器
        
        注：若del_first_line为True，则输出pandas.DataFrame没有列名
        
    Returns:
        data: list或pandas.DataFrame
    '''
	
    if logger is None:
        logger = simple_logger()
    
    if not os.path.exists(fpath):
        logger.warning('文件不存在，返回None：{}'.format(fpath))
        return None
    
    def read_lines(fapth):
        try:
            with open(fpath, 'r') as f:
                lines = f.readlines()
        except:
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            except:
                try:
                    with open(fpath, 'r', encoding='gbk') as f:
                        lines = f.readlines()
                except:
                    logger.warning(
                        '未正确识别文件编码格式，以二进制读取: {}'.format(fpath))
                    with open(fpath, 'rb') as f:
                        lines = f.readlines()
        return lines
    
    if encoding is not None:
        try:
            with open(fpath, 'r', encoding=encoding) as f:
                lines = f.readlines()
        except:
            lines = read_lines(fpath)
    else:
        lines = read_lines(fpath)    
    
    data = []
    for line in lines:
        line = str(line)
        line = line.strip()
        line = line.split(sep)
        if del_first_col:
            line = line[1:]
        data.append(line)
        
    if del_first_line:
        data = data[1:]
        if to_pd:
            data = pd.DataFrame(data)
    else:
        if to_pd:
            if keep_header:
                cols = data[0]
                data = pd.DataFrame(data[1:])
                data.columns = cols
            else:
                data = pd.DataFrame(data)
    
    return data
        
    
def load_csv(fpath, del_unname_cols=True, logger=None, encoding=None,
             **kwargs):
    '''
    用pandas读取csv数据
    
    Args:
        fpath: csv文件路径
        del_unname_cols: 是否删除未命名列，默认删除
        logger: 日志记录器
        encoding: 指定编码方式，默认不指定，不指定时会尝试以uft-8和gbk编码读取
        **kwargs: 其它pd.read_csv支持的参数
        
    Returns:
        data: pandas.DataFrame
    '''    
    
    if logger is None:
        logger = simple_logger()
    
    if not os.path.exists(fpath):
        logger.warning('文件不存在，返回None：{}!'.format(fpath))
        return None
        
    try:
        data = pd.read_csv(fpath, encoding=encoding, **kwargs)
    except:
        try:
            data = pd.read_csv(fpath, encoding='utf-8', **kwargs)
        except:
            try:
                data = pd.read_csv(fpath, encoding='gbk', **kwargs)
            except:
                data = pd.read_csv(fpath, **kwargs)
        
    if del_unname_cols:        
        del_cols = [x for x in data.columns if 'Unnamed:' in str(x)]
        if len(del_cols) > 0:
            data.drop(del_cols, axis=1, inplace=True)
            
    return data
    

if __name__ == '__main__':
    fpath = './test/load_text_test_utf8.csv'
    data1 = load_text(fpath, encoding='gbk')
    data2 = load_csv(fpath, encoding='gbk')
    