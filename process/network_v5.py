'''''''''
@file: network_v5.py
@author: MRL Liu
@time: 2020/12/18 13:03
@env: Python,Numpy
@desc:本模式从神经元层面利用矩阵向量运算实现了一个参数化的FNN：
        默认激活函数为Sigmoid
        默认损失函数为MSE（均方误差）
        默认网络优化算法为SGD（随机梯度下降）,并且添加了L2正则化
        默认梯度计算算法为BP（反向传播算法）
        默认网络初始化方式为Xavier初始化
        测试了神经网络的参数保存与重加载功能（相关方法在network_tool）
@ref:
@blog: https://blog.csdn.net/qq_41959920
'''''''''
import numpy as np
import time
import random
from matplotlib import pyplot as plt
from FNN_Work_By_Liu.process import activation_function as af
from FNN_Work_By_Liu import mnist_loader
from FNN_Work_By_Liu import network_tool as net_tool
plt.rcParams['font.sans-serif']=['SimHei'] #使用中文字符
plt.rcParams['axes.unicode_minus'] = False #显示负数的负号

"""参数化神经网络类"""
class Network(object):
    def __init__(self,shape_size):
        """shape_size是一个包含有各层神经元数量的列表"""
        self.num_layer = len(shape_size)# 神经层的数量（输入层+中间层+输出层）
        self.shape_size = shape_size # 包含有各层神经元数量的列表
        # 多个偏置向量的列表，第一层（输入层）没有偏置，所以biases[0]存储的是第二层的偏置参数列表（长度为第二层神经元数）
        self.biases = [ np.random.randn(x,1)
                        for x in shape_size[1:]]
        # 多个权重矩阵的列表，例如weights[0]表示第一层和第二层之间的权重矩阵（行数为第二层神经元数，列数为第一层神经元数）
        #self.weights = [ np.random.randn(x,y)/np.sqrt(x)
                        #for x,y in zip(shape_size[1:],shape_size[:-1])]
        self.weights = [np.random.randn(x,y)/np.sqrt((x+y)/2)
                        for x, y in zip(shape_size[1:], shape_size[:-1])]

    """输入一个多维向量，输出网络的输出"""
    def feedforward(self,x):
        for w,b in zip(self.weights,self.biases):
            x =af.Sigmoid(np.dot(w,x)+b)
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
        # 遍历每个神经层（除了输入层）的权重矩阵和偏置向量
        for w,b in zip(self.weights,self.biases):
            wx_plus_b = np.dot(w,activation)+b
            wx_plus_b_list.append(wx_plus_b)
            activation = af.Sigmoid(wx_plus_b)
            activation_list.append(activation)
        # 反向传播(从输出层开始更新神经网络的参数)
        # 计算输出层误差
        delta = self.Deriv_Loss(y_true,activation_list[-1])*\
                af.Deriv_Sigmoid(wx_plus_b_list[-1])
        # 计算输出层参数的梯度
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta,activation_list[-2].transpose())
        # 计算中间层参数的梯度
        for l in range(2,self.num_layer):
            delta =np.dot(self.weights[-l+1].transpose(),delta)* \
                   af.Deriv_Sigmoid(wx_plus_b_list[-l])
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

    """损失函数对y_pred的偏导数"""
    def Deriv_Loss(self, y_true, y_pred):
        """计算损失函数对y_pred的偏导数"""
        return (y_pred - y_true)



"""用于实验的神经网络扩展类"""
class experiment_Network(Network):
    def __init__(self, shape_size):
        super().__init__(shape_size)
        self.test_evaluate_list = [] # 训练正确率列表
        self.train_evaluate_list = [] # 测试正确率列表
        self.temp_loss_list = []
        self.loss_list = []

    """使用SGD算法训练神经网络"""
    def train_by_SGD(self, training_data, epochs, mini_batch_size, learning_rate,
                     lmbda = 0.0,test_data=None):
        """``training_data``是一个元组``(x, y)``的列表，它代表着输入和标签。
        ``epochs``是训练次数；
        ``mini_batch_size``是最小梯度批量更新的数量；
        ``learning_rate``是学习率"""

        training_data = list(training_data)  # 将training_data转换为列表
        n = len(training_data)  # 获取训练数据的长度

        if test_data:
            test_data = list(test_data)

        # 遍历进行epochs次的训练
        for i in range(epochs):
            random.shuffle(training_data)  # 将列表中的元素打乱
            # 将训练数据按照mini_batch_size划分成若干组
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate,lmbda,len(training_data))  # 使用一个最小批次来更新神经网络
                self.mini_batches_Loss(mini_batch)  # 计算并且记录均方损失

            self.epoch_Loss()
            # 测试数据
            if test_data and i%10==0:
                n_test_data = len(test_data)
                success_rate_test = self.evaluate_in_test(test_data)
                data = training_data[:1000]
                n_data = len(data)
                success_rate_train = self.evaluate_in_train(data)
                print("Epoch {} :训练集中的成功率：{}/{} 测试集中的成功率：{} / {}".format(i, success_rate_train, n_data,
                                                                         success_rate_test, n_test_data))
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
    def MSE_Loss(self, y_true, y_pred):
        inpit_n_row = len(y_true)
        result = np.sum(1 / inpit_n_row * np.square(y_true - y_pred))
        return result

    """计算每个小批次的平均损失值"""
    def mini_batches_Loss(self, data):
        cost = 0.0
        num = 0
        for x, y_true in data:
            y_pred = self.feedforward(x)  # 计算出预测值
            cost += self.MSE_Loss(y_true, y_pred)
            num += 1
        cost = cost / num
        # print(cost)
        self.temp_loss_list.append(cost)
        return cost

    """计算每个回合的平均损失值"""
    def epoch_Loss(self):
        n = len(self.temp_loss_list)
        if n != 0:
            new_loss = sum(self.temp_loss_list) / len(self.temp_loss_list)
            self.temp_loss_list.clear()
            self.loss_list.append(new_loss)






if __name__=="__main__":
    use_trained_net = False # 是否测试加载神经网络参数功能
    if use_trained_net==True:
        # 创建神经网络
        net = experiment_Network([784,30,10])
        # 创建输入数据
        training_data, validation_data, test_data = mnist_loader.load_data_wrapper('../mnist.pkl.gz')
        training_data = list(training_data)
        test_data = list(test_data)
        # 训练神经网络
        start_time = time.clock()  # 记录开始运行时间
        net.train_by_SGD(training_data[:1000], epochs=10,mini_batch_size=10, learning_rate=3.0,
                         lmbda=0.1,test_data=test_data)
        end_time = time.clock()  # 记录结束运行时间
        dtime = end_time - start_time
        print("本次实验训练共花费：{:.8f}秒 ".format(dtime))  # 记录结束运行时间
        net_tool.save_net(net, "../config/net.json") #保存神经网络参数
        e = net.test_evaluate_list

        n_loss = len(net.loss_list)
        x_data_loss = [i for i in range(n_loss)]
        y_data_loss = net.loss_list
        # 创建画布
        fig = plt.figure(figsize=(12, 4))  # 创建一个大小为（14,7）的画布

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
        y_data_evaluate_train = [data * 0.1 for data in net.train_evaluate_list]

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
        y_data_evaluate_test = [data * 0.01 for data in net.test_evaluate_list]

        # 添加第三个窗口
        ax3 = fig.add_subplot(133)  # 添加一个1行2列的序号为1的窗口
        # 添加标注
        ax3.set_title('测试集中的正确率', fontsize=14)  # 设置标题
        ax3.set_xlabel('x(训练次数)', fontsize=14, fontfamily='sans-serif', fontstyle='italic')
        ax3.set_ylabel('y(正确率%)', fontsize=14, fontstyle='oblique')
        ax3.set_ylim(0, 100)

        # 绘制函数
        ax3.plot(x_data_evaluate_test, y_data_evaluate_test, color='blue', label="正确值")
    else :
        training_data, validation_data, test_data = mnist_loader.load_data_wrapper('../mnist.pkl.gz')
        test_data = list(test_data)
        # 获取训练好的神经网络
        net = net_tool.load_net("../config/net.json")
        #net = experiment_Network([784,30,10]) # 使用一个新建的网络测试
        # 测试正确率
        n_test_data = len(test_data)
        results = [(np.argmax(net.feedforward(x)), y)
                   for (x, y) in test_data]
        success_rate_test = sum(int(x == y) for (x, y) in results)

        print("测试集中的成功率：{} / {}".format(success_rate_test, n_test_data))

