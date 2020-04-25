# coding=utf-8
"""
Data matching & categorizing module
function: data segmentation based on different seasons and different achieve rate
match&cate(merge,ar)
"""
import numpy as np
import pandas as pd
from set_path import save_path

def seg1(merge, ar):
    """
    将原始数据集按照季节、达成率进行数据切分
    :param merge: merge.csv文件, 包含所有的运行数据
    :param ar: achieve_rate.csv文件, 包含不同时间对应的达成率
    :return:
    """
    ar_time = ar['日']
    YMD = list(merge['时间'])
    HMS = list(merge['时间'])
    clean = list(merge['1#镜场清洁度'])
    clean_record = clean[0:3]
    for i in range(len(YMD)):
        ind = YMD[i].index(' ')  # 寻找原始时间戳中的空格位置
        temp = YMD[i]
        YMD[i] = temp[:ind]  # 时间格式转化为Y/M/D
        if i == 0:
            record = list(np.reshape(np.array(merge.columns.values), (1, -1)))
            temp = merge.values[i].T
            temp[0] = HMS[i]
            record.append(temp)
            continue
        elif YMD[i] != YMD[i - 1]:  # 日期变化
            for j in range(len(ar_time)):
                if YMD[i - 1] == ar_time[j]:  # 寻找上一段对应匹配的季节&效率
                    season = ar['季节'][j] + '/'
                    rate = ar['达成率水平'][j] + '/'
                    time = YMD[i - 1]
                    ind1 = time.index('/')
                    ind2 = time.index('/', ind1 + 1)
                    time = time[:ind1] + '_' + time[ind1 + 1:ind2] + '_' + time[ind2 + 1:]
                    record[1:4][-1] = clean_record  # 保存镜场清洁度数值
                    print('\t' + season + rate + time)
                    record = pd.DataFrame(record)
                    pd.DataFrame(record).to_csv(save_path + 'preprocessing/match&classify/' + season + rate + time + '.csv', header=0,
                                                index=0)
                    clean_record = clean[i:i + 3]
                    record = list(np.reshape(np.array(merge.columns.values), (1, -1)))
                    temp = merge.values[i].T
                    temp[0] = HMS[i]
                    record.append(temp)
                    break
        else:
            try:  # 当表格有数据时
                flag = sum(merge.values[i, 4:11].astype(float))  # 排除空白值情况
                temp = merge.values[i].T
                temp[0] = HMS[i]
                record.append(temp)
            except:  # 当表格无数据时
                continue
    for j in range(len(ar_time)):
        if YMD[i - 1] == ar_time[j]:  # 寻找上一段对应匹配的季节&效率
            season = ar['季节'][j] + '/'
            rate = ar['达成率水平'][j] + '/'
            time = YMD[i - 1]
            ind1 = time.index('/')
            ind2 = time.index('/', ind1 + 1)
            time = time[:ind1] + '_' + time[ind1 + 1:ind2] + '_' + time[ind2 + 1:]
            record[1:4][-1] = clean_record  # 保存镜场清洁度数值
            print('\t' + season + rate + time)
            record = pd.DataFrame(record)
            pd.DataFrame(record).to_csv(save_path + 'preprocessing/match&classify/' + season + rate + time + '.csv', header=0,
                                        index=0)


def seg2(KMeans, ar):
    """
    将KMeans聚类数据按照季节、达成率进行数据切分
    :param KMeans: KMeans_result.csv文件, 包含聚类后数据
    :param ar: achieve_rate.csv文件, 包含不同时间对应的达成率
    :return:
    """
    ar_time = ar['日']
    YMD = list(KMeans[:, 1])  # 时间信息
    HMS = list(KMeans[:, 1])
    for i in range(len(YMD)):
        ind = YMD[i].index(' ')  # 寻找原始时间戳中的空格位置
        temp = YMD[i]
        YMD[i] = temp[:ind]  # 时间格式转化为Y/M/D
        if i == 0:
            # record = list(np.reshape(np.array(KMeans.columns.values), (1, -1)))
            # temp = KMeans.values[i].T
            record = np.array(('cate', 'time'))
            temp = KMeans[i, :]
            temp[1] = HMS[i]
            record = np.vstack((record, temp))
            continue
        elif YMD[i] != YMD[i - 1]:  # 日期变化
            for j in range(len(ar_time)):
                if YMD[i - 1] == ar_time[j]:  # 寻找上一段对应匹配的季节&效率
                    season = ar['季节'][j] + '/'
                    rate = ar['达成率水平'][j] + '/'
                    time = YMD[i - 1]
                    ind1 = time.index('/')
                    ind2 = time.index('/', ind1 + 1)
                    time = time[:ind1] + '_' + time[ind1 + 1:ind2] + '_' + time[ind2 + 1:]
                    print('\t' + season + rate + time)
                    record = pd.DataFrame(record)
                    pd.DataFrame(record).to_csv(save_path + 'classifier/KMeans/' + season + rate + time + '.csv', header=0,
                                                index=0)
                    # record = list(np.reshape(np.array(KMeans.columns.values), (1, -1)))
                    # temp = KMeans.values[i].T
                    record = list(('cate', 'time'))
                    temp = KMeans[i, :]
                    temp[1] = HMS[i]
                    record = np.vstack((record, temp))
                    break
        else:
            # temp = KMeans.values[i].T
            temp = KMeans[i, :]
            temp[1] = HMS[i]
            record = np.vstack((record, temp))
    for j in range(len(ar_time)):
        if YMD[i - 1] == ar_time[j]:  # 寻找上一段对应匹配的季节&效率
            season = ar['季节'][j] + '/'
            rate = ar['达成率水平'][j] + '/'
            time = YMD[i - 1]
            ind1 = time.index('/')
            ind2 = time.index('/', ind1 + 1)
            time = time[:ind1] + '_' + time[ind1 + 1:ind2] + '_' + time[ind2 + 1:]
            print('\t' + season + rate + time)
            record = pd.DataFrame(record)
            pd.DataFrame(record).to_csv(save_path + 'classifier/KMeans/' + season + rate + time + '.csv', header=0,
                                        index=0)
