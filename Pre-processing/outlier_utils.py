# coding=utf-8
"""
*****************************
Outlier
function: data segmentation based on different seasons and different achieve rate
match&cate(merge,ar)
"""
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
import math
from set_path import read_path, save_path, file

name = pd.read_csv(read_path + 'title.csv', delimiter=',', header=0).columns.values  # 读入表格抬头

# 离群点筛查及重新赋值选项
step_forward = 5  # 拟合采用前序数据范围
step_backward = 5  # 拟合采用后序数据范围
detect_range = 100  # 检测所需的数据范围
p = 3  # 拟合曲线次数
outlier_count = 0  # 离群点计数器


def outlier_detect(data_in):
    """
    对输入数据采用IsolationForest进行离群点检测
    :param data_in: 输入待检测数据
    :return: 输入数据序列中的离群点索引（注意索引是按照输入数据来定义的，而非全序列）
    """
    # 拟合模型
    data = np.reshape(data_in, (-1, 1))
    clf = IsolationForest(n_estimators=50, contamination=0.05)  # 使用IsolationForest
    clf.fit(data)
    label = clf.predict(data)  # 为每个输入数据附上标签（-1表示离群点）
    outlier_index = np.transpose(np.array(np.where(label == -1)))  # 离群样本的索引
    outlier_index = np.reshape(outlier_index, (-1,))
    return outlier_index


def insert(data_in, nor_ind, p_ind, max_len):
    """
    利用即时学习方法对离群点进行赋值
    :param data_in: 待处理数据序列
    :param nor_ind: 序列中正常样本索引
    :param p_ind: 待处理离群点数据索引
    :param max_len: m数据序列长度
    :return: 离群点赋予的数值
    """
    data = np.array(data_in[nor_ind]).flatten()
    x = np.arange(max(0, p_ind - step_forward),
                  min(max_len, p_ind + step_backward + 1))  # define the interval for curve fitting
    x = np.array(x[nor_ind], dtype='float')
    data = np.array(data, dtype='float')
    z1 = np.polyfit(x, data, min(p, len(data)))  # use p-order polynomial to fit the data
    p1 = np.poly1d(z1)
    yvals = p1(p_ind)  # the insert value for the outlier
    return yvals


def outlier_process():
    """
    此函数通过对切分后的数据（match&classify）进行离群点筛查及重新赋值
    :return: None
    """
    ind = np.where(name == '发电量')  # 针对发电量开始值不为0的情况
    for k in range(len(file)):
        ele = file.iat[k, 0]
        print('\t' + ele)
        ori = np.array(pd.read_csv(read_path + 'preprocessing/match&classify/' + ele, delimiter=',', encoding='unicode_escape'))
        count = 0
        power = ori[:, ind]
        while power[count] != 0:
            power[count] = 0
            count = count + 1
        ori[:, ind] = power
        ori = np.array(ori)
        d = np.array(ori[:, 1:-1])  # 待处理数据（除去时间，抬头以及末列清洁度信息）
        d_new = d
        for i in range(d.shape[1]):  # 针对不同属性
            num = 0  # record the amount of outliers for specific product
            for j in range(math.ceil(d.shape[0] / detect_range)):  # 按照检测范围进行数据分段
                outlier_index = outlier_detect(d[j * detect_range:min((j + 1) * detect_range, d.shape[0]), i])
                outlier_index = outlier_index + j * detect_range  # 将子序列index转化为全序列index
                if outlier_index.size == detect_range:
                    continue
                for item in outlier_index:
                    item = int(item)
                    index_all = np.arange(max(0, item - step_forward), min(d.shape[0], item + step_backward + 1))
                    normal_index = np.setdiff1d(index_all, outlier_index)  # 范围内所有数据与离群数据的差集即为正常数据
                    normal_index = normal_index - max(0, item - step_forward)
                    if normal_index.size < (step_forward + step_backward) * 0.5:  # 当正常样本数量较少时不进行拟合
                        continue
                    num = num + 1  # 离群点计数+1
                    sub_data = d[max(0, item - step_forward):min(d.shape[0], item + step_backward + 1), i]
                    insert_value = insert(sub_data, normal_index, item - max(0, item - step_forward), d.shape[0])
                    d_new[item, i] = insert_value
        save_d = np.reshape(ori[:, 0], (-1, 1))  # 时间信息
        save_d = np.hstack((save_d, d_new))
        save_d = np.vstack((name, save_d))
        pd.DataFrame(save_d).to_csv(save_path + 'preprocessing/outlier_processed/' + ele, header=0, index=0)


def outlier_assemble():
    record = np.array(
        pd.read_csv(read_path + "preprocessing/outlier_processed/" + file.iat[0, 0], delimiter=',', header=0,
                    encoding='unicode_escape'))
    for i in range(1, len(file)):
        temp = np.array(pd.read_csv(read_path + "preprocessing/outlier_processed/" + file.iat[i, 0], delimiter=',', header=0,
                                    encoding='unicode_escape'))
        record = np.vstack((record, temp))
    record = np.vstack((name, record))
    pd.DataFrame(record).to_csv(save_path + 'preprocessing/outlier_processed/assemble.csv', header=0, index=0)
