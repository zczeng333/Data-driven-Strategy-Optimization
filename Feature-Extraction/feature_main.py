from PCA_utils import pca
from plot_utils import *
from autoencoder_utils import AutoEncoder, train_data

'''
***************************************************************
设置主程序运行参数，选择需要执行的数据处理操作
***************************************************************
'''
PCA_para = [False, 'hetero', False]  # PCA参数，0：是否进行PCA，1：PCA类型（'homo'为同质，'hetero'为异质），2：是否plot相关关系图
# AutoEncoder参数，0：是否进行AutoEncoder，1：是否准备训练数据，2：是否训练模型；3：是否利用模型进行降维，4：是否聚合数据，5：是否plot属性折线图
Auto_para = [False, False, False, False, False, False]
t_para = [3, 3, 20]  # AutoEncoder训练参数，0：滑窗尺寸，1：移动步长，2：压缩属性数

'''
***************************************************************
特征提取阶段（PCA降维/自编码器降维）
***************************************************************
'''
if PCA_para[0]:
    print('***********************' + '\n' + 'PCA...')
    pca(PCA_para[1])  # PCA可选参数'homo'/'hetero'，分别代表数据同质降维或异质降维
if PCA_para[2]:
    print('***********************' + '\n' + 'plotting...')
    plot('Cor')  # 绘制PCA降维数据的相关关系图

if Auto_para[0]:
    print('***********************' + '\n' + 'AutoEncoder...')
    A = AutoEncoder(t_para[0], t_para[1], t_para[2])  # 实例化AutoEncoder对象
    if Auto_para[1]:
        train_data()  # 准备训练数据，执行此代码时不需要
    if Auto_para[2]:
        A.train()  # 训练过程，执行此代码时不需要
    if Auto_para[3]:
        A.compress()  # 利用训练模型对数据进行降维
    if Auto_para[4]:
        A.assemble()  # 将降维后数据聚合，便于后续聚类
if Auto_para[5]:
    plot('Auto')
