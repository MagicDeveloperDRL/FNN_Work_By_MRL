'''''''''
@file: network.py
@author: MRL Liu
@time: 2020/12/3 16:50
@env: Python,Numpy
@desc: 本模块实现了一个可移植的可自定义层数和神经层数量的参数化FNN类
        默认激活函数为Sigmoid
        默认损失函数为MSE（均方误差）
        默认网络优化算法为SGD（随机梯度下降）
        默认梯度计算算法为BP（反向传播算法）
        默认网络初始化方式为高斯分布中随机采样
        
@ref:
@blog: https://blog.csdn.net/qq_41959920
'''''''''
import numpy as np
from FNN_Work_By_Liu.process import activation_function as af


class Network(object):
    def __init__(self,shape_size):
        """shape_size是一个包含有各层神经元数量的列表"""
        self.num_layers = len(shape_size) # 神经层的数量（输入层+中间层+输出层）
        self.shape_size = shape_size # 包含有各层神经元数量的列表
        # 多个偏置向量的列表，第一层（输入层）没有偏置，所以biases[0]存储的是第二层的偏置参数列表（长度为第二层神经元数）
        self.biases = [np.random.randn(x, 1)
                       for x in shape_size[1:]]
        # 多个权重矩阵的列表，例如weights[0]表示第一层和第二层之间的权重矩阵（行数为第二层神经元数，列数为第一层神经元数）
        self.weights = [np.random.randn(x,y)
                        for x,y in zip(shape_size[1:],shape_size[:-1])]


    """输入一个多维向量，输出网络的输出"""
    def feedforward(self,x):
        for w,b in zip(self.weights,self.biases):
            x = af.Sigmoid(np.dot(w,x)+b)
        return x

    """使用BP算法计算网络中每个参数对应的梯度大小"""
    def backprop(self, x, y_true):
        # 计算损失函数对相应参数的偏导数，优化后的整个神经网络的权重矩阵列表
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # 优化后的整个神经网络的偏置向量列表
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        # 前向传播
        activation = x  # 输入层的输入值直接作为激活值
        activation_list = [x]  # 存储每个神经层的输出
        wx_plus_b_list = []
        # 遍历每个神经层（除了输入层）的权重矩阵和偏置向量
        for w,b in zip(self.weights,self.biases):
            wx_plus_b = np.dot(w, activation) + b
            wx_plus_b_list.append(wx_plus_b)
            activation = af.Sigmoid(wx_plus_b)
            activation_list.append(activation)
        # 反向传播(从输出层开始更新神经网络的参数)
        # 计算输出层误差
        delta = self.Deriv_Loss(y_true, activation_list[-1]) * \
                af.Deriv_Sigmoid(wx_plus_b_list[-1])
        # 计算输出层参数的梯度
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activation_list[-2].transpose())
        # 计算中间层参数的梯度
        for l in range(2, self.num_layers):
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * \
                    af.Deriv_Sigmoid(wx_plus_b_list[-l])
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activation_list[-l-1].transpose())
        return (nabla_w, nabla_b)

    """使用mini_batch的样本对网络的参数进行梯度优化，mini_batch是一个（x，y）的样例元组的列表"""
    def update_mini_batch(self,mini_batch,learning_rate):
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
        self.weights = [w - (learning_rate / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (learning_rate / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    """使用单个样本更新网络参数"""
    def update_in_example(self, x, y_true, learning_rate):
        nabla_ws, nable_bs = self.backprop(x, y_true)
        self.weights = [w - learning_rate * nw
                        for w, nw in zip(self.weights, nabla_ws)]
        self.biases = [b - learning_rate * nb
                       for b, nb in zip(self.biases, nable_bs)]


    """损失函数对y_pred的偏导数"""
    def Deriv_Loss(self,y_true,y_pred):
        """计算损失函数对y_pred的偏导数"""
        return (y_pred - y_true)



