# coding=utf-8
import os
import pandas as pd
import numpy as np
from numpy.linalg import eig
from set_path import read_path, save_path, file

p_thres = 0.95  # 保留主成分时的CPV阈值
attr_num = np.loadtxt('attr_col1.txt', dtype=int)  # for different classes of attributes


def pca_cal(x):
    """
    principal component analysis
    :param x: 待进行PCA降维的数据矩阵
    :return:
    """
    ave = np.mean(x, axis=0)
    x = x - ave
    cov = np.dot(x.T, x) / (x.shape[0] - 1)  # covariance of input data
    val, vec = eig(cov)  # eigenvalues & eigenvectors
    p = np.zeros((1, len(val))).reshape((-1, 1))  # record percent variance for each component
    weight = np.zeros((x.shape[1], len(val)))  # record projection
    for i in range(len(val)):
        index = np.argmax(val)  # weight with the highest variance
        p[i] = val[index]
        weight[:, i] = vec[:, index]
        val[index] = 0
    return p, np.dot(x, weight)


def hetero_pc():
    """
    不同天数据采取不同的PCA提取标准，依据CPV筛选主成分
    """
    for i in range(len(file)):  # 依次读入不同季节不同达成率的数据表格
        ele = file.iat[i, 0]
        x = pd.read_csv(read_path + 'preprocessing/outlier_processed/' + ele, delimiter=',', engine='python')
        x = np.array(x)[:, :-1]
        PC = np.zeros((x.shape[0] + 1, 1)).astype(str)
        PC[0, 0] = 'Percent'  # 第一行行名Percent, 即PC对应百分比
        PC[1:, 0] = x[:, 0]
        for i in range(len(attr_num) - 1):
            x_new = x[:, attr_num[i]:attr_num[i + 1]].astype(float)
            per, pc = pca_cal(x_new)
            per = per / sum(per)
            temp1 = np.vstack((per.T, pc))
            temp2 = np.empty((temp1.shape[0], 0))
            count = 0
            for j in range(len(per)):
                if count > p_thres:  # PC累计百分比达到一定的数量
                    break
                else:
                    temp2 = np.hstack((temp2, np.reshape(temp1[:, j], (-1, 1))))
                    count = count + temp1[0, j]
            PC = np.hstack((PC, temp2))
        ind = ele.index('.')
        np.savetxt(save_path + 'feature/PCA/hetero_standard/selected/' + ele[:ind] + '_pca.csv', PC, fmt='%s',
                   delimiter=',')


def homo_pc():
    """
    所有数据采取相同的PCA提取标准，依据CPV筛选主成分
    提取两次主成分，第一次attri→class，第二次class→大类
    """
    inte = pd.read_csv(read_path + 'preprocessing/outlier_processed/' + file.iat[0, 0], delimiter=',', engine='python',
                       header=0)
    inte = inte.dropna(axis=0, how='any')
    YMD = list(inte['时间'])
    ar = pd.read_csv(read_path + 'achieving_rate.csv', delimiter=',', header=0)
    ar_time = ar['日']
    inte = np.array(inte)
    for i in range(1, len(file)):  # 整合所有数据
        temp = pd.read_csv(read_path + 'preprocessing/outlier_processed/' + file.iat[i, 0], delimiter=',',
                           engine='python')
        temp = temp.dropna(axis=0, how='any')
        YMD = YMD + list(temp['时间'])
        inte = np.vstack((inte, np.array(temp)))
    PC1 = np.reshape(inte[:, 0], (-1, 1))  # 时间信息
    percent1 = np.empty((1, 0))  # 累计百分比信息
    for i in range(len(attr_num) - 1):  # 依据不同大类分别进行pca_cal处理
        x_new = inte[:, attr_num[i]:attr_num[i + 1]].astype(float)
        per1, pc1 = pca_cal(x_new)
        per1 = per1 / sum(per1)
        count = 0
        for j in range(len(per1)):
            if count > p_thres:  # PC1累计百分比达到一定的数量
                break
            else:
                PC1 = np.hstack((PC1, np.reshape(pc1[:, j], (-1, 1))))  # 主成分信息
                percent1 = np.hstack((percent1, np.reshape(per1[j], (1, 1))))
                count = count + per1[j]
    x = np.vstack((np.zeros((1, PC1.shape[1])), PC1))
    x[0, 1:] = percent1
    x[0, 0] = 'Percent'
    np.savetxt(save_path + 'feature/PCA/homo_standard/assemble.csv', x, fmt='%s', delimiter=',')
    x = np.zeros((1, PC1.shape[1]))
    x[0, 1:] = percent1
    for i in range(1, len(YMD)):
        ind = YMD[i].index(' ')  # 寻找原始时间戳中的空格位置
        temp = YMD[i]
        YMD[i] = temp[:ind]  # 时间格式转化为Y/M/D
        if YMD[i] != YMD[i - 1]:  # 日期变化
            for j in range(len(ar_time)):
                if YMD[i - 1] == ar_time[j]:  # 寻找上一段对应匹配的季节&效率
                    season = ar['季节'][j] + '/'
                    rate = ar['达成率水平'][j] + '/'
                    time = YMD[i - 1]
                    ind1 = time.index('/')
                    ind2 = time.index('/', ind1 + 1)
                    time = time[:ind1] + '_' + time[ind1 + 1:ind2] + '_' + time[ind2 + 1:]
                    if not os.path.exists(save_path + 'feature/PCA/homo_standard/' + season + rate):
                        os.makedirs(save_path + 'feature/PCA/homo_standard/' + season + rate)
                    x[0, 0] = 'Percent'
                    np.savetxt(save_path + 'feature/PCA/homo_standard/' + season + rate + time + '.csv', x, fmt='%s',
                               delimiter=',')
                    x = np.zeros((1, PC1.shape[1]))
                    x[0, 1:] = percent1
                    break
        else:
            x = np.vstack((x, PC1[i, :]))
    for j in range(len(ar_time)):
        if YMD[i - 1] == ar_time[j]:  # 寻找上一段对应匹配的季节&效率
            season = ar['季节'][j] + '/'
            rate = ar['达成率水平'][j] + '/'
            time = YMD[i - 1]
            ind1 = time.index('/')
            ind2 = time.index('/', ind1 + 1)
            if not os.path.exists(save_path + 'feature/PCA/homo_standard/' + season + rate):
                os.makedirs(save_path + 'feature/PCA/homo_standard/' + season + rate)
            x[0, 0] = 'Percent'
            np.savetxt(save_path + 'feature/PCA/homo_standard/' + season + rate + time + '.csv', x, fmt='%s',
                       delimiter=',')
            x = np.zeros((1, PC1.shape[1]))
            x[0, 1:] = percent1
            break


def pca(select='homo'):
    if select == 'homo':
        print('PCA type: homo-standard')
        homo_pc()
    elif select == 'hetero':
        print('PCA type: hetero-standard')
        hetero_pc()
    else:
        raise TypeError('Unrecognized PCA type')
