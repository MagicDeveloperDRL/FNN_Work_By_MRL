'''''''''
@file: run_experiment_1.py
@author: MRL Liu
@time: 2020/12/3 17:00
@env: Python,Numpy
@desc: 本模块用来使用network类进行函数拟合的实验
@ref:
@blog: https://blog.csdn.net/qq_41959920
'''''''''
import numpy as np
import random
#from FNN_Work_By_Liu import network
from DeepLearningPython35 import network
from matplotlib import pyplot as plt

"""本实验的神经网络类"""
class experiment_1_Network(network.Network):

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
                rand_num = np.random.randint(0,9)*0.01
                print("Epoch {} : 逼近值：{:.8f} / {:.8f}  测试值：{:.8f}/{:.8f} 准确率：{}/{}".format(i,
                       Renormalization(self.feedforward(0.009))[0][0], Target_Function(0.9),
                       Renormalization(self.feedforward(rand_num))[0][0],Target_Function(rand_num),
                       self.evaluate(test_data), n_test))
            else:
                print("Epoch {} complete".format(i))


    def evaluate(self, test_data):
        """返回测试数据中的正确率"""
        test_results = [(Renormalization(self.feedforward(x)[0][0]), y)
                        for (x, y) in test_data]
        return sum(int(np.abs(y-x)<=0.5) for (x, y) in test_results)

"""要拟合的目标函数"""
def Target_Function(x, a=1.0, b=1.0, c=1.0, d=1.0):
    y = a * np.sin(b * x) + c * np.cos(d * x)
    # y = a*b*x*c*d
    return y

"""归一化函数（将数据限制到0-1之间）"""
def Normalization(data):
    global N_Min
    N_Min = np.min(data)
    print(N_Min)
    global N_Range
    N_Range = np.max(data) - N_Min
    print(N_Range)
    return (data - N_Min) / N_Range

"""反归一化函数（）"""
def Renormalization(x):
    return x * N_Range + N_Min

"""构建数据集"""
def get_DataSet(start = 0,stop = 2,step = 50,show_dataSet=True):
    # 构建样本和标签
    global x_data
    x_data = np.linspace(start, stop, step)  # 从0-2之间均匀采样的50个样本
    y_data = Target_Function(x_data)
    #print(y_data[0])
    y_data = Normalization(y_data) # 归一化
    #print(y_data[0])
    if show_dataSet:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(x_data, y_data)
        plt.ion()
        plt.show()
    # 定义输入数据
    dataSet = [(x, y) for x,y in zip(x_data,y_data) ]
    #y_data = Normalization([dataSet[i][1] for i in step])
    #dataSet = np.array(dataSet)
    return dataSet
def show_DataSet(start = 0,stop = 2,step = 50):
    # 构建样本和标签
    global x_data
    y_data = range(0,len(x_data))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x_data, y_data)
    plt.ion()
    plt.show()
if __name__ =="__main__":
    # 创建神经网络
    net = experiment_1_Network([1, 3, 3, 1])
    # y = Target_Function(x=1.0)
    # print(y)
    # 定义输入数据
    training_data = get_DataSet(-10, 10, 500)
    #test_data = get_DataSet1(10, 20, 100)
    # test_data = get_DataSet(3, 5, 50)
    # print(training_data.shape)
    # 训练神经网络
    #net.train_by_SGD(training_data=training_data, epochs=50000, mini_batch_size=10, learning_rate=1,
                    # test_data=test_data)
    # 测试
    # print(Target_Function(x=1.0))
    # print(net.feedforward(1.0))