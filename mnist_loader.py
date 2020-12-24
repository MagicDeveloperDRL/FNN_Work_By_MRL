'''''''''
@file: mnist_loader.py
@author: MRL Liu
@time: 2020/12/3 17:01
@env: Python,Numpy
@desc: 本模块提供加载Mnist.pkl.gz格式的数据集，并对数据集进行处理的方法
@ref:
@blog: https://blog.csdn.net/qq_41959920
'''''''''
import pickle
import gzip
import numpy as np

"""加载Mnist.pkl.gz数据集"""
def load_mnist(filepath='mnist.pkl.gz'):
    # 从当前文件夹中读取文件数据
    f = gzip.open(filepath, 'rb')
    # 从数据中获取Python对象训练集、验证集、测试集
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    # 关闭文件读取
    f.close()
    return (training_data, validation_data, test_data)

"""加载转换形式后的数据集"""
def load_data_wrapper(filepath='mnist.pkl.gz'):
    tr_d, va_d, te_d = load_mnist(filepath) # 获取训练集、验证集、测试集
    # 构建新的训练集
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]] # 遍历训练集中的样本，并将样本转换成（784,1）的形状
    training_results = [vectorized_result(y) for y in tr_d[1]] # 遍历训练集中的标签，并将标签转换为（10,1）的形状
    training_data = zip(training_inputs, training_results) # 将测试样本和训练标签共同组成新的训练集
    # 构建新的验证集
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    # 构建新的测试集
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

"""转换标签"""
def vectorized_result(j):
    e = np.zeros((10, 1)) # 生成一个十行一列的列表
    e[j] = 1.0
    return e
