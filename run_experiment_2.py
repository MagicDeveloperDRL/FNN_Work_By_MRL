'''''''''
@file: run_experiment_2.py
@author: MRL Liu
@time: 2020/12/3 17:01
@env: Python,Numpy
@desc:
@ref:
@blog: https://blog.csdn.net/qq_41959920
'''''''''
import numpy as np
import pickle
import gzip
import random
from FNN_Work_By_Liu import network
#from DeepLearningPython35 import network
from FNN_Work_By_Liu import mnist_loader
"""本实验的神经网络类"""
class experiment_2_Network(network.Network):

    # 使用训练数据按照随机梯度下降算法训练神经网络
    def train_by_SGD(self, training_data, epochs, mini_batch_size, learning_rate,
            test_data=None):
        """``training_data``是一个元组``(x, y)``的列表，它代表着输入和标签。
        ``epochs``是训练次数；
        ``mini_batch_size``是最小梯度批量更新的数量；
        ``learning_rate``是学习率"""

        training_data = list(training_data) # 将training_data转换为列表
        n = len(training_data) #获取训练数据的长度

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        # 遍历进行epochs次的训练
        for i in range(epochs):
            random.shuffle(training_data) # 将列表中的元素打乱
            # 将训练数据按照mini_batch_size划分成若干组
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate) # 使用一个最小批次来更新神经网络
            # 测试数据
            if test_data:
                print("Epoch {} : 测试成功率：{} / {}".format(i, self.evaluate(test_data), n_test));
            else:
                print("Epoch {} complete".format(i))


    def evaluate(self, test_data):
        """返回测试数据中的正确率"""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)



if __name__=="__main__":
    # 创建神经网络
    net = experiment_2_Network([784, 30, 10])
    # 创建输入数据
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    training_data = list(training_data)
    # 训练神经网络
    net.train_by_SGD(training_data, 30, 10, 3.0, test_data=test_data)
