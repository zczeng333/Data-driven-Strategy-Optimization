import numpy as np
from sklearn.cluster import DBSCAN
from set_path import save_path


def dbscan(d_in, type="Unweighted", save=True):
    # Prepare data for DBScan
    data = np.array(d_in)[:, 1:]  # 去除时间数据
    if type == 'Weighted':  # 有权聚类
        weight = np.array(d_in.columns.values[1:]).astype(float)
    elif type == 'Unweighted':  # 无权聚类
        weight = np.ones((1, data.shape[1])).reshape(-1)
    else:
        raise TypeError('Unrecognized clustering type')
    nor_data = normal(data)  # normalize feature value

    # DBScan clustering,eps: radius of scanning circle, min_samples: min number of samples required in each circle
    db = DBSCAN(eps=0.7, min_samples=10).fit(weight * nor_data)
    result = db.labels_
    d_out = np.hstack((np.zeros((np.array(d_in).shape[0], 1)), np.array(d_in)))  # space to store labels
    d_out = np.vstack((np.zeros((1, d_out.shape[1])), d_out))  # place to store category index
    d_out[0, 0] = 'Cate'
    d_out[0, 1:] = d_in.columns.values
    d_out[1:, 0] = np.transpose(result)
    if save:
        np.savetxt(save_path + 'classifier/DBScan_result.csv', d_out, fmt='%s', delimiter=',')


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
