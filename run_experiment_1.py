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
import time
from FNN_Work_By_Liu import network
#from DeepLearningPython35 import network
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #使用中文字符
plt.rcParams['axes.unicode_minus'] = False #显示负数的负号

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
            if test_data and i%100==0:
                rand_num = np.random.randint(0,9)*0.01
                print("Epoch {} : 逼近值：{:.8f} / {:.8f}  测试值：{:.8f}/{:.8f} 准确率：{}/{}".format(i,
                       Renormalization(self.feedforward(0.009))[0][0], Target_Function(0.9),
                       Renormalization(self.feedforward(rand_num))[0][0],Target_Function(rand_num),
                       self.evaluate(test_data), n_test))
            elif i%100==0:
                print("Epoch {} complete".format(i))


    def evaluate(self, test_data):
        """返回测试数据中的正确率"""
        test_results = [(renormalize(self.feedforward(x)[0][0]), y)
                        for (x, y) in test_data]
        return sum(int(np.abs(y-x)<=0.5) for (x, y) in test_results)

"""要拟合的目标函数"""
def Target_Function(x, a=1.0, b=1.0, c=1.0, d=1.0,n=10):
    #y = 0.0
    #for i in range(n):
        #y += a * np.sin(b * x) + c * np.sin(d * x)
    y = a * np.sin(b * x) + c * np.cos(d * x)
    # y = a*b*c*d*x
    return y

"""归一化函数（将数据限制到0-1之间）"""
def normalize(data):
    data_min, data_max = data.min(), data.max()
    data = (data - data_min) / (data_max - data_min)
    return data
"""反归一化函数（）"""
def renormalize(norm_data, data):
    data_min, data_max = data.min(), data.max()
    return norm_data * (data_max - data_min) + data_min


# 归一化数据集(包含数据和标签)
def normalize_dataset(dataset):
    xs = np.array([data[0] for data in dataset])
    xs = normalize(xs)
    ys = np.array([data[1] for data in dataset])
    ys = normalize(ys)
    nor_dataset = [ (x,y) for x,y in zip(xs,ys)]
    return nor_dataset
def renormalize_dataset(norm_dataset,dataset):
    xs = np.array([data[0] for data in dataset])
    xs = renormalize(xs)
    ys = np.array([data[1] for data in dataset])
    ys = renormalize(ys)
    nor_dataset = [ (x,y) for x,y in zip(xs,ys)]
    return nor_dataset

"""构建数据集"""
def get_DataSet(start = 0,stop = 2,step = 50):
    # 构建样本和标签
    x_data = np.linspace(start, stop, step)  # 从0-2之间均匀采样的50个样本
    y_data = Target_Function(x_data)
    #print(y_data[0])
    #x_data = normalize(x_data)
    y_data = normalize(y_data) # 归一化
    #print(y_data[0])
    # 定义输入数据
    dataSet = [(x, y) for x,y in zip(x_data,y_data) ]
    #y_data = Normalization([dataSet[i][1] for i in step])
    #dataSet = np.array(dataSet)
    return dataSet


if __name__ =="__main__":
    # 创建神经网络
    net = experiment_1_Network([1, 3, 3, 1])
    # 获取训练集和测试集
    training_data = get_DataSet(-10, 10, 500)
    test_data = get_DataSet(10, 20, 100)
    # print(training_data.shape)
    # 训练神经网络
    start_time = time.clock() # 记录开始运行时间
    net.train_by_SGD(training_data=training_data, epochs=50000, mini_batch_size=20, learning_rate=1,
                     )
    end_time = time.clock() # 记录结束运行时间
    dtime = end_time - start_time
    print("本次实验训练共花费：{:.8f}秒 ".format(dtime)) # 记录结束运行时间
    # 获取要绘制的数据
    x_data = [data[0] for data in training_data]
    y_true_data = [data[1] for data in training_data]
    y_pre_data = [net.feedforward(x)[0][0] for x in x_data]

    # 创建画布
    fig = plt.figure(figsize=(14, 7))

    # 绘制第一个窗口
    ax = fig.add_subplot(121)  # 创建一个画布和一个绘制对象（画布大小为（14,7））
    # 添加标注
    ax.set_title('训练集中的函数拟合', fontsize=18)  # 设置标题
    ax.set_xlabel('x-自变量', fontsize=18, fontfamily='sans-serif', fontstyle='italic')
    ax.set_ylabel('y-因变量', fontsize='x-large', fontstyle='oblique')
    # 绘制真实函数
    line1, = ax.plot(x_data, y_true_data, color='blue', label="真实值", linestyle='--')
    # 绘制模拟函数
    line2,=ax.plot(x_data, y_pre_data, color='red',label="预测值")
    ax.legend(handles=[line1,line2], loc=2)  # 绘制图例说明

    # 获取要绘制的数据
    x_data_test = [data[0] for data in test_data]
    y_true_data_test = [data[1] for data in test_data]
    y_pre_data_test = [net.feedforward(x)[0][0] for x in x_data_test]
    # 绘制第二个窗口
    ax = fig.add_subplot(122)  # 创建一个画布和一个绘制对象（画布大小为（14,7））
    # 添加标注
    ax.set_title('测试集中的函数拟合', fontsize=18)  # 设置标题
    ax.set_xlabel('x-自变量', fontsize=18, fontfamily='sans-serif', fontstyle='italic')
    ax.set_ylabel('y-因变量', fontsize='x-large', fontstyle='oblique')
    # 绘制真实函数
    line1, = ax.plot(x_data_test, y_true_data_test, color='blue', label="真实值", linestyle='--')
    # 绘制模拟函数
    line2, = ax.plot(x_data_test, y_pre_data_test, color='red', label="预测值")
    ax.legend(handles=[line1, line2], loc=2)  # 绘制图例说明