'''''''''
@file: network_tool.py
@author: MRL Liu
@time: 2020/12/19 9:56
@env: Python,Numpy
@desc:本模式用来实现一些神经网络的使用工具：
        （1）Sigmoid函数及其导数
        （2）保存和读取神经网络参数的函数
        （3）平方差损失函数、交叉熵损失函数
@ref:
@blog: https://blog.csdn.net/qq_41959920
'''''''''
import numpy as np
from enum import Enum
import json
from FNN_Work_By_Liu.process.network_v5 import Network

# Sigmoid激活函数: f(x) = 1 / (1 + e^(-x))
def Sigmoid(x):
    return 1.0/(1.0+np.exp(-x))
# Sigmoid激活函数的导数: f'(x) = e^(-x) / (1 + e^(-x))^2=f(x)*(1-f(x))
def Deriv_Sigmoid(x):
    fx = Sigmoid(x)
    return fx*(1-fx)

"""保存神经网络的参数,filename的形式为'../config/record.json'。"""
def save_net(network,filename):
    data = {"sizes": network.shape_size,
            "weights": [w.tolist() for w in network.weights],
            "biases": [b.tolist() for b in network.biases]}
    f = open(filename, "w")
    json.dump(data, f)#将Python数据结构编码为JSON格式并且保存至文件中
    f.close()#关闭文件
    print("神经网络参数成功保存至{}文件".format(filename))

"""读取神经网络的参数"""
def load_net(filename,net=None):
    f = open(filename, "r")
    data = json.load(f)#将文件中的JSON格式解码为Python数据结构
    f.close()
    if net==None:
        net = Network(data["sizes"])
        net.weights = [np.array(w) for w in data["weights"]]
        net.biases = [np.array(b) for b in data["biases"]]
        return net
    else:
        net.weights = [np.array(w) for w in data["weights"]]
        net.biases = [np.array(b) for b in data["biases"]]
        return

"""均方损失函数（MSE）"""
class MSE_Loss(object):
    """计算损失"""
    @staticmethod
    def loss(y_true, y_pred):
        return 0.5 * np.linalg.norm(y_pred - y_true) ** 2
        #inpit_n_row = len(y_true)
        #result = np.sum(1 / inpit_n_row * np.square(y_true - y_pred))
        #return result

    """计算损失的偏导数"""
    @staticmethod
    def deriv(wx_plus_b,y_true, y_pred):
        return y_pred - y_true

    """计算输出层的误差"""
    @staticmethod
    def delta(wx_plus_b,y_true, y_pred):
        return MSE_Loss.deriv(wx_plus_b,y_true, y_pred)* Deriv_Sigmoid(wx_plus_b)
"""交叉熵损失函数"""
class CrossEntropy_Loss(object):

    """计算损失"""
    @staticmethod
    def loss(y_true, y_pred):
        result = np.sum(np.nan_to_num(-y_true*np.log(y_pred)-(1-y_true)*np.log(1-y_pred)))
        return result

    """计算损失的偏导数"""
    @staticmethod
    def deriv(wx_plus_b,y_true, y_pred):
        return (y_pred - y_true)/Deriv_Sigmoid(wx_plus_b)

    """计算输出层的误差"""
    @staticmethod
    def delta(wx_plus_b,y_true, y_pred):
        return (y_pred - y_true)

"""网络参数初始器"""
class Parameter_Initializer(object):
    def __init__(self,network,initializer_Type):
        self.initializer_type = initializer_Type
        self.Init(network,initializer_Type)

    @staticmethod  # 参数初始化
    def Init(network,initializer_Type,filepath=None):
        if initializer_Type == Parameter_Initializer.type.RANDOM:
            Parameter_Initializer.Random_initializer(network)
        elif initializer_Type == Parameter_Initializer.type.XAVIER:
            Parameter_Initializer.Xavier_initializer(network)
        elif initializer_Type == Parameter_Initializer.type.LOAD:
            Parameter_Initializer.Load_initializer(network,filepath)

    @staticmethod # 随机初始化
    def Random_initializer(network):
        # 多个权重矩阵的列表，例如weights[0]表示第一层和第二层之间的权重矩阵（行数为第二层神经元数，列数为第一层神经元数）
        network.weights = [np.random.randn(y, x)
                        for x, y in zip(network.shape_size[:-1], network.shape_size[1:])]
        # 多个偏置向量的列表，第一层（输入层）没有偏置，所以biases[0]存储的是第二层的偏置参数列表（长度为第二层神经元数）
        network.biases = [np.random.randn(y, 1)
                       for y in network.shape_size[1:]]

    @staticmethod # Xavier初始化
    def Xavier_initializer(network):
        # 多个权重矩阵的列表，例如weights[0]表示第一层和第二层之间的权重矩阵（行数为第二层神经元数，列数为第一层神经元数）
        network.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(network.shape_size[:-1], network.shape_size[1:])]
        # 多个偏置向量的列表，第一层（输入层）没有偏置，所以biases[0]存储的是第二层的偏置参数列表（长度为第二层神经元数）
        network.biases = [np.random.randn(y, 1)
                       for y in network.shape_size[1:]]

    @staticmethod  # 导入参数初始化
    def Load_initializer(network,filepath):
        load_net(filepath,network)

    # 初始化方式的枚举类
    class type(Enum):
        RANDOM = 0
        XAVIER = 1
        LOAD = 2



