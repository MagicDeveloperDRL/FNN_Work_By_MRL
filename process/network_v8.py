'''''''''
@file: network_v8.py
@author: MRL Liu
@time: 2020/12/26 17:25
@env: Python,Numpy
@desc:本模式从神经层角度利用矩阵向量运算实现了一个参数化的FNN：
        （1）网络初始化方式参数化，提供随机初始化、Xavier初始化、导入之前参数等多种方式
        （2）默认激活函数为Sigmoid,输出层是否激活可自由选择。
        （3）损失函数参数化，可以选择MSE损失函数、交叉熵损失函数
        （4）默认梯度计算算法为BP（反向传播算法）
        （5）提供网络优化算法为SGD（随机梯度下降）,并且添加了L2正则化
@ref:
@blog: https://blog.csdn.net/qq_41959920
'''''''''


import time
import random
import signal
from matplotlib import pyplot as plt
import numpy as np
from FNN_Work_By_Liu.network_tool import *
from FNN_Work_By_Liu import mnist_loader

plt.rcParams['font.sans-serif']=['SimHei'] #使用中文字符
plt.rcParams['axes.unicode_minus'] = False #显示负数的负号

Net_Parameter_Save_Path = "../config/net.json"  # 网络参数保存路径
Net_Parameter_Load_Path = "../config/net.json"  # 网络参数加载路径

"""参数化神经网络类"""
class Network(object):
    def __init__(self,shape_size,initializer_type,loss_function,filepath=None,activate_out=True):
        """shape_size是一个包含有各层神经元数量的列表"""
        self.num_layer = len(shape_size)# 神经层的数量（输入层+中间层+输出层）
        self.shape_size = shape_size # 包含有各层神经元数量的列表
        self.initializer_type = initializer_type # 参数初始化方式
        Parameter_Initializer.Init(self,initializer_type,filepath=filepath) # 初始化参数
        self.loss_function = loss_function # 损失函数
        self.activate_out =activate_out # 输出层是否需要激活函数
    """输入一个多维向量，输出网络的输出"""
    def feedforward(self,x):
        if self.activate_out:
            for w,b in zip(self.weights,self.biases):
                x =Sigmoid(np.dot(w,x)+b)
        else:
            # 中间层使用激活函数
            for w, b in zip(self.weights[:-1], self.biases[:-1]):
                x = Sigmoid(np.dot(w, x) + b)
            # 输出层不使用激活函数
            x = np.dot(self.weights[-1], x) + self.biases[-1]
        return x

    """使用BP算法计算网络中每个参数对应的梯度大小"""
    def backprop(self,x,y_true):
        # 计算损失函数对相应参数的偏导数，优化后的整个神经网络的权重矩阵列表
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # 优化后的整个神经网络的偏置向量列表
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        # 前向传播
        activation = x  # 输入层的输入值直接作为激活值
        activation_list =[x] # 存储每个神经层的输出
        wx_plus_b_list = []
        if self.activate_out:
            # 遍历中间层（除了输入层）的权重矩阵和偏置向量
            for w, b in zip(self.weights, self.biases):
                wx_plus_b = np.dot(w, activation) + b
                wx_plus_b_list.append(wx_plus_b)
                activation = Sigmoid(wx_plus_b)
                activation_list.append(activation)
        else:
            # 遍历中间层（除了输入层）的权重矩阵和偏置向量
            for w,b in zip(self.weights[:-1],self.biases[:-1]):
                wx_plus_b = np.dot(w,activation)+b
                wx_plus_b_list.append(wx_plus_b)
                activation = Sigmoid(wx_plus_b)
                activation_list.append(activation)
            # 计算输出层
            wx_plus_b = np.dot(self.weights[-1], activation) + self.biases[-1]
            wx_plus_b_list.append(wx_plus_b)
            activation_list.append(wx_plus_b)
        # 反向传播(从输出层开始更新神经网络的参数)
        # 计算输出层误差
        if self.activate_out:
            delta = (self.loss_function).delta(wx_plus_b=wx_plus_b_list[-1], y_pred=activation_list[-1], y_true=y_true)
        else:
            delta = (self.loss_function).deriv(wx_plus_b=wx_plus_b_list[-1], y_pred=activation_list[-1], y_true=y_true)
        # 计算输出层参数的梯度
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta,activation_list[-2].transpose())
        # 计算中间层参数的梯度
        for l in range(2,self.num_layer):
            delta =np.dot(self.weights[-l+1].transpose(),delta)* \
                   Deriv_Sigmoid(wx_plus_b_list[-l])
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta,activation_list[-l-1].transpose())
        return (nabla_w,nabla_b)

    """使用mini_batch的样本更新网络参数，
    mini_batch是一个（x，y）的样例元组的列表
    lmbda是一个正则化参数
    n是训练数据的大小"""
    def update_mini_batch(self,mini_batch,learning_rate,lmbda,n):
        # 存储本批次的整个神经网络的权重对应的梯度的累计和
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # 存储本批次的整个神经网络的偏置对应的梯度的累计和
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        for x,y in mini_batch:
            # 存储本次计算后的每个参数的梯度
            delta_nabla_w,  delta_nabla_b= self.backprop(x, y)
            # 将本次计算的梯度累加到之前计算的梯度
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # 利用本批次求得的各个参数的梯度平均值更新神经网络的权重和偏置参数
        self.weights = [(1-learning_rate*(lmbda/n))*w - (learning_rate / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (learning_rate / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]









