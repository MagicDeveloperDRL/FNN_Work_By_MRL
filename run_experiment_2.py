'''''''''
@file: run_experiment_2.py
@author: MRL Liu
@time: 2020/12/3 17:01
@env: Python,Numpy
@desc: 本模块用来使用network类进行手写数字识别分类的实验
@ref:
@blog: https://blog.csdn.net/qq_41959920
'''''''''
import numpy as np
from matplotlib import pyplot as plt
import random
from FNN_Work_By_Liu import network
from FNN_Work_By_Liu import mnist_loader
#from DeepLearningPython35 import network


plt.rcParams['font.sans-serif']=['SimHei'] #使用中文字符
plt.rcParams['axes.unicode_minus'] = False #显示负数的负号

"""本实验的神经网络类"""
class experiment_2_Network(network.Network):
    def __init__(self, shape_size):
        super().__init__(shape_size)
        self.test_evaluate_list = [] # 训练正确率列表
        self.train_evaluate_list = [] # 测试正确率列表
        self.temp_loss_list = []
        self.loss_list = []

    """使用SGD算法训练神经网络"""
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
                self.mini_batches_Loss(mini_batch)  # 计算并且记录均方损失

            self.epoch_Loss()
            # 测试数据
            if test_data :
                success_rate_test=self.evaluate_in_test(test_data[:100])
                success_rate_train = self.evaluate_in_train(training_data[:100])
                print("Epoch {} :训练集中的成功率：{}/{} 测试集中的成功率：{} / {}".format(i, success_rate_train,100,success_rate_test, 100))
                self.test_evaluate_list.append(success_rate_test)
                self.train_evaluate_list.append(success_rate_train)
            #else:
                #print("Epoch {} complete".format(i))

    """使用GD算法训练神经网络"""
    def train_by_GD(self, training_data, epochs, learning_rate,
                     test_data=None):
        """``training_data``是一个元组``(x, y)``的列表，它代表着输入和标签。
        ``epochs``是训练次数；
        ``mini_batch_size``是最小梯度批量更新的数量；
        ``learning_rate``是学习率"""

        training_data = list(training_data)  # 将training_data转换为列表
        n = len(training_data)  # 获取训练数据的长度

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        # 遍历进行epochs次的训练
        for i in range(epochs):
            random.shuffle(training_data)  # 将列表中的元素打乱
            # 将训练数据按照mini_batch_size划分成若干组
            for x,y in training_data:
                self.update_in_example(x,y, learning_rate)  # 使用一个最小批次来更新神经网络

            # 计算每个回合的损失

            self.mini_batches_Loss(training_data[0])  # 计算并且记录均方损失
            self.epoch_Loss()
            # 测试数据
            if test_data:
                success_rate_test = self.evaluate_in_test(test_data[:100])
                success_rate_train = self.evaluate_in_train(training_data[:100])
                print("Epoch {} :训练集中的成功率：{}/{} 测试集中的成功率：{} / {}".format(i, success_rate_train, 100, success_rate_test,
                                                                         100))
                self.test_evaluate_list.append(success_rate_test)
                self.train_evaluate_list.append(success_rate_train)
            # else:
            # print("Epoch {} complete".format(i))
    """计算每个回合测试的正确率"""
    def evaluate_in_test(self, data):
        """返回测试数据中的正确率"""
        results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)
    def evaluate_in_train(self, data):
        """返回测试数据中的正确率"""
        results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                        for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    """均方损失函数（MSE），并记录损失值"""
    def MSE_Loss(self,y_true,y_pred):
        inpit_n_row = len(y_true)
        result = np.sum(1 / inpit_n_row * np.square(y_true - y_pred))
        return result

    """计算每个小批次的平均损失值"""
    def mini_batches_Loss(self, data):
        cost = 0.0
        num =0
        for x, y_true in data:
            y_pred = self.feedforward(x) # 计算出预测值
            cost+=self.MSE_Loss(y_true,y_pred)
            num +=1
        cost = cost/num
        #print(cost)
        self.temp_loss_list.append(cost)
        return cost

    """计算每个回合的平均损失值"""
    def epoch_Loss(self):
        n = len(self.temp_loss_list)
        if  n!=0:
            new_loss =sum(self.temp_loss_list)/len(self.temp_loss_list)
            self.temp_loss_list.clear()
            self.loss_list.append(new_loss)




if __name__=="__main__":
    # 创建神经网络
    net = experiment_2_Network([784, 30,30, 10])
    # 创建输入数据
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    training_data = list(training_data)
    test_data = list(test_data)
    # 训练神经网络
    #net.train_by_SGD(training_data[:1000], 100, 10, 3.0, test_data=test_data)
    net.train_by_GD(training_data[:1000], 100,  3.0, test_data=test_data)
    e=net.test_evaluate_list

    n_loss = len(net.loss_list)
    x_data_loss = [i for i in range(n_loss)]
    y_data_loss = net.loss_list

    # 创建画布
    fig = plt.figure(figsize=(12, 4)) # 创建一个大小为（14,7）的画布

    # 添加第一个窗口
    ax1 = fig.add_subplot(131)  # 添加一个1行2列的序号为1的窗口
    # 添加标注
    ax1.set_title('数字识别实验的损失变化图', fontsize=14)  # 设置标题
    ax1.set_xlabel('x(训练次数)', fontsize=14, fontfamily='sans-serif', fontstyle='italic')
    ax1.set_ylabel('y(损失大小)', fontsize=14, fontstyle='oblique')
    ax1.set_ylim(0, 1)
    # 绘制真实函数
    ax1.plot(x_data_loss, y_data_loss, color='red', label="损失值")

    # 获取要绘制的数据
    n_evaluate_train = len(net.train_evaluate_list)
    x_data_evaluate_train = [i for i in range(n_evaluate_train)]
    y_data_evaluate_train = net.train_evaluate_list

    # 添加第二个窗口
    ax2 = fig.add_subplot(132)  # 添加一个1行2列的序号为1的窗口
    # 添加标注
    ax2.set_title('训练集中的正确率', fontsize=14)  # 设置标题
    ax2.set_xlabel('x(训练次数)', fontsize=14, fontfamily='sans-serif', fontstyle='italic')
    ax2.set_ylabel('y(正确率%)', fontsize=14, fontstyle='oblique')
    ax2.set_ylim(0, 100)

    # 绘制函数
    ax2.plot(x_data_evaluate_train, y_data_evaluate_train, color='blue', label="正确值")

    # 获取要绘制的数据
    n_evaluate_test = len(net.test_evaluate_list)
    x_data_evaluate_test = [i for i in range(n_evaluate_test)]
    y_data_evaluate_test = net.test_evaluate_list

    # 添加第三个窗口
    ax3 = fig.add_subplot(133)  # 添加一个1行2列的序号为1的窗口
    # 添加标注
    ax3.set_title('测试集中的正确率', fontsize=14)  # 设置标题
    ax3.set_xlabel('x(训练次数)', fontsize=14, fontfamily='sans-serif', fontstyle='italic')
    ax3.set_ylabel('y(正确率%)', fontsize=14, fontstyle='oblique')
    ax3.set_ylim(0, 100)

    # 绘制函数
    ax3.plot(x_data_evaluate_test, y_data_evaluate_test, color='blue', label="正确值")
