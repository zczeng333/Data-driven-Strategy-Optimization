"""
This python file implements the following function:
implements kmeans clustering utils for the project
normal(): Apply z-score normalization to three features in 'product_features.csv'
dis(): calculate Euclidean distance between two vectors
randCent(): create k center randomly
kMeans(): Apply K-means clustering to all products(1000) based on 'attribute1','attribute2' and 'original price'
clu_plot(): Plot the clustering result
save_cate_data(): save data into different category folders based on the clustering result
clustering(): combine the above functions and form an executable function
"""
import numpy as np
import pandas as pd
from math import *
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from set_path import save_path

c_num = 3  # number of classes


def normal(d_in):
    """
    this function implements the following function:
    normalization process for different attributes (z-score)
    :param d_in: input data [[x11,x12,...],...,[xn1,xn2,...]]
    :return:d_out: normalized data
    """
    d_out = np.empty(shape=(d_in.shape[0], 0))
    for i in range(d_in.shape[1]):
        temp = np.reshape(d_in[:, i], (-1, 1))
        mean = np.mean(temp)
        std = np.std(temp)
        temp = (temp - mean) / std
        d_out = np.hstack((d_out, temp))
    return d_out


def dist(A, B, w):
    """
    this function implements the following function:
    calculate euclidean distance between A and B
    :param A: vector A
    :param B: vector B
    :param w: weight vector
    :return: dis (float)
    """
    return np.sqrt(sum(np.power(w * (A - B), 2)))


def randCent(d_in, k):
    """
    this function implements the following function:
    create k center randomly
    :param d_in: input data
    :param k: clustering number
    :return: centroids [[c11,...,c1n],...,[ck1,...,ckn]]
    """
    d_in = np.array(d_in, dtype=np.float)
    centroids = np.empty(shape=(0, d_in.shape[1]), dtype=np.float)
    record = np.array([])
    for i in range(k):
        flag = True
        while flag:
            index = random.randint(0, d_in.shape[0] - 1)
            flag = bool(np.size(np.where(record == index)) != 0)  # random point already exists?
        record = np.hstack((record, index))
        centroids = np.vstack((centroids, d_in[index]))
    return centroids


def kMeans(d_in, w, k):
    """
    this function implements the following function:
    K-means clustering algorithm
    :param d_in: input data [[x1,y1,...],[x2,y2,...],....,[xn,yn,...]]
    :param k: clustering number
    :return: centroids[[class 1, pos 2],[class 2, pos 2]...], result[[point 1, dis 1], [point2, dis2],...]
    """
    m = np.shape(d_in)[0]
    result = np.zeros((m, 2))  # store the clustering result [[class1,dis1],...,[classn,disn]]
    centroids = randCent(d_in, k)  # generate random centers for different classes
    flag = True  # to judge whether the clustering process has converged
    count = 1
    while flag:
        flag = False
        print('\t' + 'iteration: %s' % count)
        for i in range(m):  # assign points to their nearest class
            min_dist = inf
            min_ind = -1
            for j in range(k):
                distJI = dist(centroids[j, :], d_in[i, :], w)
                if distJI < min_dist:
                    min_dist = distJI
                    min_ind = j
            if result[i, 0] != min_ind:
                flag = True  # if change exists, continue iteration process
            result[i, :] = min_ind, min_dist ** 2
        for cent in range(k):  # recalculate center points for different classes
            ptsInClust = d_in[np.nonzero(result[:, 0] == cent)[0]]
            centroids[cent, :] = np.mean(ptsInClust, axis=0)
        count = count + 1
    return centroids, result


def clu_plot(cate, cor):
    """
    this function implements the following function:
    save the clustering result figure as 'figure/clustering_result.png'
    :param cate:
    :param cor:
    :return:
    """
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(cor[:, 0], cor[:, 1], cor[:, 2], c=cate)
    ax.set_zlabel('original price', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('attribute 2', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('attribute 1', fontdict={'size': 15, 'color': 'red'})
    plt.show()
    plt.savefig('figure/clustering_result.png')


def kmeans_clustering(d_in, type='Unweighted', num=c_num, save=True):
    """
    this function implements the following function:
    combine the above functions and form an executable function
    :return: None
    """
    data = np.array(d_in)[:, 1:]
    if type == 'Weighted':  # 有权聚类
        weight = np.array(d_in.columns.values[1:]).astype(float)
    elif type == 'Unweighted':  # 无权聚类
        weight = np.ones((1, data.shape[1])).reshape(-1)
    else:
        raise TypeError('Unrecognized clustering type')
    nor_fea = normal(data)  # normalize feature value
    myCentroids, result = kMeans(nor_fea, weight, num)
    d_out = np.hstack((np.zeros((np.array(d_in).shape[0], 1)), np.array(d_in)))
    d_out = np.vstack((np.zeros((1, d_out.shape[1])), d_out))
    d_out[0, 0] = 'Cate'
    d_out[0, 1:] = d_in.columns.values
    d_out[1:, 0] = result[:, 0]
    if save:
        # np.savetxt(save_path + 'classifier/KMeans_result.csv', d_out, fmt='%s', delimiter=',')
        pd.DataFrame(d_out).to_csv(save_path + 'classifier/KMeans_result.csv', header=0, index=0)
        print("result saved in " + save_path + 'classifier/KMeans_result.csv')
    return myCentroids, result


def best_k(d_in):
    record = inf
    bestk = 0
    weight = np.array(d_in.columns.values[1:]).astype(float)
    for k in range(2, 7):
        c, r = kmeans_clustering(d_in, save=False, num=k)  # 在k个类别下进行聚类
        sum = 0
        for i in range(k):
            for j in range(i, k):
                sum = sum + dist(c[i, :], c[j, :], weight)
        r_out = 2 * sum / (k * (k - 1))  # 类间距离（希望max）
        r_in = np.sum(np.sqrt(r[:, :]))  # 类内距离（希望min）
        judge = r_in / r_out
        if record > judge:
            record = judge
            bestk = k
        print('dis for k=%d is %f, ' % (k, judge) + 'best dis is %f, ' % record + 'best k is %d' % bestk)
    print('best k in interval [2,7] is %d' % bestk)
    return bestk
