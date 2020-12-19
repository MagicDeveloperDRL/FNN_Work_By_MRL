'''''''''
@file: network_tool.py
@author: MRL Liu
@time: 2020/12/19 9:56
@env: Python,Numpy
@desc:本模式用来实现一些神经网络的使用工具：
        保存神经网络参数的函数
        读取神经网络参数的函数
@ref:
@blog: https://blog.csdn.net/qq_41959920
'''''''''
import numpy as np
import sys
import json
import network_v5

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
def load_net(filename):
    f = open(filename, "r")
    data = json.load(f)#将文件中的JSON格式解码为Python数据结构
    f.close()
    net = network_v5.Network(data["sizes"])
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net