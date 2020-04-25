from kmeans_utils import kmeans_clustering, best_k
from dbscan_utils import dbscan
from plot_utils import *
from segmentation_utils import *
from set_path import read_path
from RandomForest import rf
import pandas as pd
import numpy as np

'''
***************************************************************
设置主程序运行参数，选择需要执行的数据处理操作
***************************************************************
'''
# 聚类参数，0：是否进行聚类，1：聚类类型（'KMeans', 'GMM', 'DBScan'），2：是否对聚类结果按ar进行切分，3：是否plot聚类结果
cluster_para = [False, 'KMeans', False, False]
KMeans_para = [False, 'weather', 5]  # KMeans聚类参数，0：是否寻找最优聚类数，1：针对哪个数据进行聚类（'Auto'/'PCA'/'weather'）2：聚类数
RM = True  # 是否采用随机森林分类器

'''
***************************************************************
读取必要数据文件
***************************************************************
'''
ar = pd.read_csv(read_path + 'achieving_rate.csv', delimiter=',', header=0)

'''
***************************************************************
无监督聚类（KMeans聚类 / DBScan聚类）
***************************************************************
'''
if cluster_para[0]:
    print('***********************' + '\n' + 'clustering...')
    if cluster_para[1] == 'KMeans':  # KMeans聚类
        if KMeans_para[1] == 'Auto':
            assemble = pd.read_csv(read_path + 'feature/AutoEncoder/assemble.csv', delimiter=',', header=None)
            if KMeans_para[0]:  # 寻找最优K值
                print('***********************' + '\n' + 'Searching for best k...')
                best_k(assemble)
            kmeans_clustering(assemble, 'Unweighted', KMeans_para[2])  # AutoEncoder结果聚类
            if cluster_para[2]:  # 对数据进行切分
                KMeans_result = np.array(
                    pd.read_csv(read_path + 'classifier/KMeans/result(AutoEncoder).csv', delimiter=',', header=0))[:,
                                :2]
                seg2(KMeans_result, ar)  # 对聚类结果进行切割，记得将相应的KMeans_result移动至data_backup/KMeans_result文件夹下
        elif KMeans_para[1] == 'PCA':
            assemble = pd.read_csv(read_path + 'feature/PCA/homo_standard/assemble.csv', delimiter=',', header=0)
            if KMeans_para[0]:
                print('***********************' + '\n' + 'Searching for best k...')
                best_k(assemble)
            kmeans_clustering(assemble, 'Weighted', KMeans_para[2])  # PCA结果聚类
            if cluster_para[2]:  # 对数据进行切分
                KMeans_result = np.array(
                    pd.read_csv(read_path + 'classifier/KMeans/result(PCA).csv', delimiter=',', header=0))[:, :2]
                seg2(KMeans_result, ar)  # 对聚类结果进行切割，记得将相应的KMeans_result移动至data_backup/KMeans_result文件夹下
        elif KMeans_para[1] == 'weather':
            data = pd.read_csv(read_path + 'preprocessing/outlier_processed/assemble.csv', delimiter=',', header=0,
                               encoding='unicode_escape').ix[:, :4]
            if KMeans_para[0]:
                print('***********************' + '\n' + 'Searching for best k...')
                best_k(data)
            kmeans_clustering(data, 'Unweighted', KMeans_para[2])  # PCA结果聚类
            if cluster_para[2]:  # 对数据进行切分
                KMeans_result = np.array(
                    pd.read_csv(read_path + 'classifier/KMeans/result(weather).csv', delimiter=',', header=0))[:, :2]
                seg2(KMeans_result, ar)  # 对聚类结果进行切割，记得将相应的KMeans_result移动至data_backup/KMeans_result文件夹下
        else:
            raise TypeError('Please specify KMeans_para[1] to be either Auto or PCA')
        if cluster_para[3]:  # plot聚类结果
            plot('KMeans')  # 对聚类结果按天进行plot，记得将相应切分后文件夹移动至data_backup/KMeans_result文件夹下
    elif cluster_para[1] == 'GMM':
        pass
        # print('***********************' + '\n' + 'clustering...')
        # c = GMM(np.array(assemble)[:, 1:], 4)
        # c.gmm_clustering()
    elif cluster_para[1] == 'DBScan':
        assemble = pd.read_csv(read_path + 'feature/AutoEncoder/assemble.csv', delimiter=',', header=None)
        dbscan(assemble, "Unweighted")
    else:
        raise TypeError('Please specify cluster_para[1] to be one of KMeans, GMM and DBScan')

'''
***************************************************************
有监督分类（随机森林）
***************************************************************
'''
if RM:
    rf()
