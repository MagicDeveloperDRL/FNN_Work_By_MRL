'''''''''
@file: run_mnist_classification_experiment.py
@author: MRL Liu
@time: 2020/12/3 17:01
@env: Python,Numpy
@desc: 本模块用来使用network类进行数字分类的实验
    
@ref:
@blog: https://blog.csdn.net/qq_41959920
'''''''''
import random
import time
from matplotlib import pyplot as plt
from FNN_Work_By_Liu import mnist_loader
from FNN_Work_By_Liu.network_tool import *
from FNN_Work_By_Liu.process.network_v7 import Network

plt.rcParams['font.sans-serif']=['SimHei'] #使用中文字符
plt.rcParams['axes.unicode_minus'] = False #显示负数的负号
Net_Parameter_Save_Path = "config/classification_net.json" # 网络参数保存路径
Net_Parameter_Load_Path = "config/classification_net.json" #

"""用于分类实验的神经网络扩展类"""
class classification_experiment_Network(Network):
    def __init__(self, shape_size,
                 initializer_type=Parameter_Initializer.type.XAVIER,
                 loss_function = MSE_Loss,filepath=None):
        super().__init__(shape_size,initializer_type,loss_function,filepath)

        self.train_loss_list = []
        self.train_accuracy_list = [] # 测试正确率列表
        self.test_loss_list = []
        self.test_accuracy_list = []  # 训练正确率列表


    """使用SGD算法训练神经网络"""
    def train_by_SGD(self, training_data, #元组(x, y)的列表，代表着输入和标签。
                     epochs, # 训练次数
                     mini_batch_size, # 最小梯度批量更新的数量；
                     learning_rate,# 学习率
                     lmbda = 0.0, # 正则化参数
                     test_data=None,# 每回合的测试数据集
                     early_stopping_n=0, # 提前终止机制的最大无效回合数
                     store_training_loss_and_accuracy=True,
                     store_test_loss_and_accuracy=True
                     ):

        training_data = list(training_data)  # 将training_data转换为列表
        n_training_data = len(training_data)  # 获取训练数据的长度
        # 测试数据集的设置:
        if test_data:
            test_data = list(test_data)
            n_test_data = len(test_data)
        # 提前终止机制的设置:
        if early_stopping_n >0:
            best_test_accuracy = 0  # 最好的正确率
            num_useless_epoch= 0  # 训练无效的回合数
        start_time = time.clock()  # 记录开始运行时间
        # 遍历进行epochs次的训练
        for epoch in range(epochs):
            random.shuffle(training_data)  # 将列表中的元素打乱
            # 将训练数据按照mini_batch_size划分成若干组
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n_training_data, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate,lmbda,len(training_data))  # 使用一个最小批次来更新神经网络
            print("第{}回合训练结束".format(epoch))
            # 如果存储训练过程的损失和精度
            if store_training_loss_and_accuracy:
                train_loss = self.get_loss(lmbda,training_data,is_test=False)
                self.train_loss_list.append(train_loss)
                train_accuracy = self.get_accuracy(training_data,is_test=False)
                self.train_accuracy_list.append(train_accuracy*100/n_training_data)
                print("训练集上的损失为{},正确率为{}/{}".format(train_loss,train_accuracy,n_training_data))
            # 如果存储测试过程的损失和精度
            if store_test_loss_and_accuracy and test_data:
                test_loss = self.get_loss(lmbda,test_data,is_test=True)
                self.test_loss_list.append(test_loss)
                test_accuracy = self.get_accuracy(test_data,is_test=True)
                self.test_accuracy_list.append(test_accuracy*100/n_test_data)
                print("测试集上的损失为{},正确率为{}/{}".format(test_loss,test_accuracy,n_test_data))
            # 提前终止机制
            if early_stopping_n > 0 and store_test_loss_and_accuracy:
                # 检查本回合的测试正确率是否有提升
                if test_accuracy > best_test_accuracy:
                    best_test_accuracy = test_accuracy
                    num_useless_epoch = 0
                else:
                    num_useless_epoch += 1
                # 如果测试正确率在指定回合数内无提升则结束训练
                if (num_useless_epoch == early_stopping_n):
                    print("已有{}次回合无法提升正确率，提前终止机制启动，训练结束".format(num_useless_epoch))
                    save_net(self, Net_Parameter_Save_Path)  # 保存本次训练参数
                    return

        end_time = time.clock()  # 记录结束运行时间
        dtime = end_time - start_time
        print("本次训练共花费：{:.8f}秒 ".format(dtime))  # 记录结束运行时间
        # 保存本次训练参数
        print("训练正常结束，正在保存网络参数...")
        save_net(self, Net_Parameter_Save_Path)
        print("网络参数已经保存至:{}".format(Net_Parameter_Save_Path))




    """计算数据集上的正确率"""
    def get_accuracy(self,data,is_test=False):
        if is_test:
            results = [(np.argmax(self.feedforward(x)), y)
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    """计算每个回合的平均损失值"""
    def get_loss(self,lmbda,data,is_test=False):
        loss = 0.0
        for x,y_true in data:
            y_pred = self.feedforward(x)
            if is_test:y_true = vectorized_lable(y_true)
            loss +=self.loss_function.loss(y_true,y_pred)/len(data)
        loss +=0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights)
        return loss

    """绘制每个回合的损失值和正确率的变化趋势"""
    def draw_loss_and_accuracy_plot(self):
        # 创建画布
        fig = plt.figure(figsize=(12, 6))  # 创建一个指定大小的画布

        # 添加第1个窗口
        ax1 = fig.add_subplot(121)  # 添加一个1行2列的序号为1的窗口
        # 添加标注
        ax1.set_title('训练中的损失变化', fontsize=14)  # 设置标题
        ax1.set_xlabel('x(训练次数)', fontsize=14, fontfamily='sans-serif', fontstyle='italic')
        ax1.set_ylabel('y(损失大小)', fontsize=14, fontstyle='oblique')
        ax1.set_ylim(0, 1)
        # 绘制函数
        n_loss = len(self.train_loss_list)
        line1, = ax1.plot(range(1, n_loss + 1,1), self.train_loss_list, color='blue', label="训练集")
        line2, = ax1.plot(range(1, n_loss + 1,1), self.test_loss_list, color='red', label="测试集")
        ax1.legend(handles=[line1, line2], loc=2)  # 绘制图例说明
        #plt.grid(True)#启用表格
        # 添加第2个窗口
        ax2 = fig.add_subplot(122)  # 添加一个1行2列的序号为1的窗口
        # 添加标注
        ax2.set_title('训练中的正确率变化', fontsize=14)  # 设置标题
        ax2.set_xlabel('x(训练次数)', fontsize=14, fontfamily='sans-serif', fontstyle='italic')
        ax2.set_ylabel('y(正确率%)', fontsize=14, fontstyle='oblique')
        ax2.set_ylim(0, 100)
        # 获取要绘制的数据
        n_accuracy = len(self.train_accuracy_list)  # 训练集和测试集的正确个数都为回合数
        # 绘制函数
        line1, = ax2.plot(range(1, n_accuracy + 1), self.train_accuracy_list, color='blue', label="训练集")
        # 绘制函数
        line2, = ax2.plot(range(1, n_accuracy + 1), self.test_accuracy_list, color='red', label="测试集")
        ax2.legend(handles=[line1, line2], loc=4)  # 绘制图例说明
        #plt.grid(True) #启用表格

"""将标签向量化"""
def vectorized_lable(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


if __name__=="__main__":
    # 创建神经网络
    net = classification_experiment_Network(shape_size=[784, 30, 10],
                                            initializer_type=Parameter_Initializer.type.XAVIER,
                                            loss_function=CrossEntropy_Loss,
                                            filepath=Net_Parameter_Load_Path)
    # 创建输入数据
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    training_data = list(training_data)
    test_data = list(test_data)
    # 训练神经网络
    net.train_by_SGD(training_data[:10000], epochs=30, mini_batch_size=10, learning_rate=0.5,
                     lmbda=0.1, test_data=test_data, early_stopping_n=10,
                     store_test_loss_and_accuracy=False,
                     store_training_loss_and_accuracy=False
                     )
    # 绘制图像
    net.draw_loss_and_accuracy_plot()