# -*- coding: utf-8 -*-
"""
@author: zzc93
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import time
from set_path import read_path, save_path

read_path = read_path + 'classifier/RandomForest/'
save_path = save_path + 'classifier/RandomForest/'


def rf():
    start = time.clock()  # 记录初始时间
    # 导入训练数据集
    data = pd.read_csv(read_path + "train_data.csv")
    train_data = data.iloc[:, 1:26]  # 去除时间数据及label
    train_labels = data.iloc[:, 26:31]  # 类别标签
    # 导入测试数据集
    predict_data = pd.read_csv(read_path + "predict_data.csv").iloc[:, 1:26]  # 去除时间数据

    # Normalization
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train_scaled = min_max_scaler.fit_transform(train_data)
    X_predict_scaled = min_max_scaler.fit_transform(predict_data)
    print('Normalization complete!')
    class_name = ['Light', 'Absorption', 'Storage', 'Generation', 'Weather']

    for i in range(5):  # 针对各类别进行预测
        # 特征选择
        print('training for class: ' + class_name[i])
        labels = train_labels.iloc[:, i]
        svc = SVC(kernel="linear")
        dt = DecisionTreeClassifier()
        rfecv = RFECV(estimator=dt, step=1, cv=StratifiedKFold(2), scoring='accuracy')
        rfecv.fit(train_data, labels)
        print("Optimal number of features : %d" % rfecv.n_features_)
        print("Ranking of features names: %s" % train_data.columns[rfecv.ranking_ - 1])
        print("Ranking of features nums: %s" % rfecv.ranking_)
        print('Feature selection complete!')


        # 使用随机森林分类器
        rf_clf = RandomForestClassifier(max_depth=100, n_estimators=3000, max_leaf_nodes=1500, oob_score=True,
                                        random_state=30,
                                        n_jobs=-1)
        rf_clf.fit(train_data, labels)
        y_predict = rf_clf.predict(predict_data)
        np.savetxt(save_path + 'result/' + class_name[i] + '.csv', y_predict, fmt='%s', delimiter=',')
        print(rf_clf.oob_score_)
        print('Random forest classifier complete!')

    # 计算运行时间
    elapsed = (time.clock() - start)
    print("Time used:", elapsed)
