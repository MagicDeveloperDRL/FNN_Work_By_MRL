'''''''''
@file: activation_function.py
@author: MRL Liu
@time: 2020/12/3 16:49
@env: Python,Numpy
@desc:本模块存放一些共有的、可移植的神经元激活函数
@ref:
@blog: https://blog.csdn.net/qq_41959920
'''''''''

import numpy as np

# Sigmoid激活函数: f(x) = 1 / (1 + e^(-x))
def Sigmoid(x):
    return 1.0/(1.0+np.exp(-x))
# Sigmoid激活函数的导数: f'(x) = e^(-x) / (1 + e^(-x))^2=f(x)*(1-f(x))
def Deriv_Sigmoid(x):
    fx = Sigmoid(x)
    return fx*(1-fx)

def Relu(x):
    return np.maximum(0, x)

def Deriv_Relu(x):
    return -np.minimum(0, x)

def Tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x)+np.exp(-x))