# coding=utf-8
"""
Data ploting module
function: plot different figures
draw(x,y,t,num,root)
correlation(data)
plot(merge,select)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.pylab import style
import os
from set_path import read_path, save_path, file

# 设置为中文字体
style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def draw(x, y, y_label, num, root):
    """
    绘制并保存折线图子函数
    :param x: 数据x轴数值
    :param y: 数据y轴数值
    :param y_label: 时间数据
    :param num: 变量编号
    :param root: 路径
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('time')
    ax.set_ylabel(y_label)
    ax.plot(x, y, c='k', marker='.')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d-%H'))
    plt.savefig(save_path + 'Figure/' + root + '/' + str(num) + '_' + y_label + '.png')
    # # 显示
    plt.close()


def correlation(data):
    """
    对输入数据矩阵绘制并保存相关关系热力图
    :param data:进行相关关系分析的数据矩阵
    :return:None
    """
    names = data.columns.values
    cor = data.corr()
    correction = abs(cor)  # 取绝对值，只看相关程度 ，不关心正相关还是负相关
    data = np.array(data)
    time = data[0, 0]
    ind1 = time.index(' ')  # 寻找原始时间戳中的空格位置
    time = time[:ind1]
    ind2 = time.index('/')  # 寻找原始时间戳中的斜杠位置
    ind3 = time[ind2 + 1:].index('/')
    save_name = time[:ind2] + '-' + time[ind2 + 1:ind2 + ind3 + 1] + '-' + time[ind2 + ind3 + 2:]
    print(save_name)
    # plot correlation matrix
    fig = plt.figure()
    plt.rcParams['font.sans-serif'] = ['Arial']
    ax = sns.heatmap(correction, cmap=plt.cm.Blues, square=True, xticklabels=False, yticklabels=False)
    plt.savefig(save_path + 'Figure/correlation/' + save_name + '.png', dpi=300)
    plt.close()


def plot(select='All'):
    """
    绘制PCA变量对应连线图
    :param merge: merge.csv文件
    :param select: All→绘制原始变量, PCA→绘制PCA降维变量, Auto->绘制AutoEncoder降维变量, Cor->绘制相关关系图, Kmeans->绘制聚类图
    :return:
    """
    merge = pd.read_csv(read_path + 'merge.csv', delimiter=',', header=1)
    title = np.array(merge.columns.values)
    if select == 'All':
        print('plotting type: all attributes')
        # 绘制原始数据所有变量采用以下代码
        for k in range(len(file)):
            ele = file.iat[k, 0]
            print('\t' + ele)
            # print(ele)
            if not os.path.exists(save_path + 'Figure/Initial/' + ele):
                os.makedirs(save_path + 'Figure/Initial/' + ele)
            data = np.array(pd.read_csv(read_path + 'preprocessing/match&classify/' + ele, delimiter=',',
                                        encoding='unicode_escape'))[
                   :, :-1]
            # 读入除去镜场清洁度以外的数据
            time = data[:, 0]
            x = [datetime.strptime(d, '%Y/%m/%d %H:%M') for d in time]
            ind1 = time[0].index(' ')  # 寻找原始时间戳中的空格位置
            ind2 = time[0].index(':')  # 寻找原始时间戳中的冒号位置
            temp1 = time[0][ind1:ind2]
            for i in range(1, len(time)):
                ind1 = time[i].index(' ')  # 寻找原始时间戳中的空格位置
                ind2 = time[i].index(':')  # 寻找原始时间戳中的冒号位置
                temp2 = time[i][ind1 + 1:ind2]
                if temp1 != temp2:  # 是否跨越1h
                    temp1 = temp2
            for j in range(1, data.shape[1]):
                y = data[:, j]
                draw(x, y, title[j], j, 'Initial/' + ele)

    elif select == 'PCA':
        print('plotting type: PCA')
        # 绘制主成分变量采用以下代码
        for k in range(len(file)):
            ele = file.iat[k, 0]
            print('\t' + ele)
            if not os.path.exists(save_path + 'Figure/PCA/' + ele):
                os.makedirs(save_path + 'Figure/PCA/' + ele)
            data = np.array(pd.read_csv(read_path + 'feature/PCA/homo_standard/' + ele, delimiter=',', engine='python'))
            time = data[:, 0]
            x = [datetime.strptime(d, '%Y/%m/%d %H:%M') for d in time]
            ind1 = time[0].index(' ')  # 寻找原始时间戳中的空格位置
            ind2 = time[0].index(':')  # 寻找原始时间戳中的冒号位置
            temp1 = time[0][ind1:ind2]
            for i in range(1, len(time)):
                ind1 = time[i].index(' ')  # 寻找原始时间戳中的空格位置
                ind2 = time[i].index(':')  # 寻找原始时间戳中的冒号位置
                temp2 = time[i][ind1 + 1:ind2]
                if temp1 != temp2:  # 是否跨越1h
                    temp1 = temp2
            for j in range(1, data.shape[1]):
                y = data[:, j]
                draw(x, y, 'attr', j, 'PCA/' + ele)

    elif select == 'Auto':
        print('plotting type: AutoEncoder')
        for k in range(len(file)):
            ele = file.iat[k, 0]
            print('\t' + ele)
            if not os.path.exists(save_path + 'Figure/AutoEncoder/' + ele):
                os.makedirs(save_path + 'Figure/AutoEncoder/' + ele)
            data = np.array(
                pd.read_csv(read_path + 'feature/AutoEncoder/Example/' + ele, delimiter=',', engine='python',
                            header=None))
            time = data[:, 0]
            x = [datetime.strptime(d, '%Y/%m/%d %H:%M') for d in time]
            ind1 = time[0].index(' ')  # 寻找原始时间戳中的空格位置
            ind2 = time[0].index(':')  # 寻找原始时间戳中的冒号位置
            temp1 = time[0][ind1:ind2]
            for i in range(1, len(time)):
                ind1 = time[i].index(' ')  # 寻找原始时间戳中的空格位置
                ind2 = time[i].index(':')  # 寻找原始时间戳中的冒号位置
                temp2 = time[i][ind1 + 1:ind2]
                if temp1 != temp2:  # 是否跨越1h
                    temp1 = temp2
            for j in range(1, data.shape[1]):
                y = data[:, j]
                draw(x, y, 'attr', j, 'AutoEncoder/' + ele)

    elif select == 'KMeans':
        print('plotting type: KMeans')
        if not os.path.exists(save_path + 'Figure/Kmeans/'):
            os.makedirs(save_path + 'Figure/KMeans/')
        for k in range(len(file)):
            ele = file.iat[k, 0]
            ind = ele.index('.')
            t = ele[:ind]
            t = t.replace('/', '_')
            print('\t' + ele)
            data = np.array(
                pd.read_csv(read_path + 'classifier/KMeans/' + ele, delimiter=',', engine='python', header=0))
            time = data[:, 1]
            x = [datetime.strptime(d, '%Y/%m/%d %H:%M') for d in time]
            ind1 = time[0].index(' ')  # 寻找原始时间戳中的空格位置
            ind2 = time[0].index(':')  # 寻找原始时间戳中的冒号位置
            temp1 = time[0][ind1:ind2]
            for i in range(1, len(time)):
                ind1 = time[i].index(' ')  # 寻找原始时间戳中的空格位置
                ind2 = time[i].index(':')  # 寻找原始时间戳中的冒号位置
                temp2 = time[i][ind1 + 1:ind2]
                if temp1 != temp2:  # 是否跨越1h
                    temp1 = temp2

            y = data[:, 0]  # category
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.set_xlabel('time')
            ax.set_ylabel('Condition')
            ax.plot(x, y, c='k', marker='.')
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d-%H'))
            plt.savefig(save_path + 'Figure/KMeans/' + t + '.png')
            plt.close()

    elif select == 'Cor':
        print('plotting type: correlation heatmap')
        # 绘制主成分变量采用以下代码
        for i in range(len(file)):
            ele = file.iat[i, 0]
            print('\t' + ele)
            if not os.path.exists(save_path + 'Figure/correlation'):
                os.makedirs(save_path + 'Figure/correlation')
            data = pd.read_csv(read_path + 'preprocessing/outlier_processed/' + ele, delimiter=',', engine='python')
            correlation(data)
    else:
        raise TypeError('Unrecognized plot type')
