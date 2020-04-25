from segmentation_utils import *
from outlier_utils import outlier_process, outlier_assemble
from set_path import read_path
import pandas as pd

'''
***************************************************************
设置主程序运行参数，选择需要执行的数据处理操作
***************************************************************
'''
Pre_para = [False, False, True]  # 预处理参数，0：是否对原始数据进行匹配与切分，1：是否对数据进行离群点筛查，2：离群点数据是否合并

'''
***************************************************************
读取必要数据文件
***************************************************************
'''
ar = pd.read_csv(read_path + 'achieving_rate.csv', delimiter=',', header=0)

'''
***************************************************************
数据预处理阶段（数据清洗、分段、离群点筛查）
***************************************************************
'''
if Pre_para[0]:
    print('***********************' + '\n' + 'matching & classifying...')
    merge = pd.read_csv(read_path + 'merge.csv', delimiter=',', header=1)
    seg1(merge, ar)
if Pre_para[1]:
    print('***********************' + '\n' + 'outlier processing...')
    outlier_process()
if Pre_para[2]:
    outlier_assemble()
