'''''''''
@file: network_v1.py
@author: MRL Liu
@time: 2020/12/16 19:34
@env: Python,Numpy
@desc: 本模式从神经元层面利用矩阵向量运算实现了一个2*2*1的FNN：
        默认激活函数为Sigmoid
        默认损失函数为MSE（均方误差）
        默认网络优化算法为GD（梯度下降）
        默认梯度计算算法为BP（反向传播算法）
        默认网络初始化方式为高斯分布中随机采样
@ref:
@blog: https://blog.csdn.net/qq_41959920
'''''''''
import numpy as np

from FNN_Work_By_Liu.process import activation_function as af

"""固定式的矩阵向量运算的神经网络类"""
class Network(object):
    def __init__(self):
        # 权重
        self.weights_layer_1 = np.random.randn(2,2) # 中间层的权重矩阵，相当于w1,w2，w3,w4
        self.weights_layer_2 = np.random.randn(1,2) # 输出层的权重矩阵，相当于w5,w6
        # 偏置
        self.biases_layer_1 = np.random.randn(2,1) # 中间层的偏置矩阵，b1，b2
        self.biases_layer_2 = np.random.randn(1,1)  # 中间层的偏置矩阵，相当于b3
    """输入一个多维向量，输出网络的输出"""
    def feedforward(self,x):
        """前向传播得到神经网络的输出"""
        # 计算第一层神经元的输出
        sum_layer_1 = self.weights_layer_1*x+self.biases_layer_1
        out_layer_1 = af.Sigmoid(sum_layer_1)
        # 计算第二层神经元的输出
        sum_layer_2 = self.weights_layer_2 * out_layer_1+self.biases_layer_2
        out_layer_2 = af.Sigmoid(sum_layer_2)
        return out_layer_2

    """使用BP算法计算网络中每个参数对应的梯度大小"""
    def backprop(self,x,y_true):
        """反向传播优化神经网络"""
        # 计算损失函数对相应参数的偏导数，优化后的整个神经网络的权重矩阵列表
        nabla_weight_layer_1 = np.zeros(self.weights_layer_1.shape)
        nabla_weight_layer_2 = np.zeros(self.weights_layer_2.shape)
        # 优化后的整个神经网络的偏置向量列表
        nabla_biase_layer_1 = np.zeros(self.biases_layer_1.shape)
        nabla_biase_layer_2 = np.zeros(self.biases_layer_2.shape)
        # 前向传播
        activation = x  # 输入层的输出，也是中间层第一层的输入
        activation_list = [x]  # 存储每个神经层的输出
        wx_plus_b_list = [] # 存储每个神经层未激活的值
        # 将第一层神经元的输出中间结果存储进列表
        wx_plus_b = np.dot(self.weights_layer_1, activation) + self.biases_layer_1
        wx_plus_b_list.append(wx_plus_b)
        activation = af.Sigmoid(wx_plus_b)
        activation_list.append(activation)
        # 将第二层神经元的输出中间结果存储进列表
        wx_plus_b = np.dot(self.weights_layer_2, activation) + self.biases_layer_2
        wx_plus_b_list.append(wx_plus_b)
        activation = af.Sigmoid(wx_plus_b)
        activation_list.append(activation)
        # 反向传播(从输出层开始计算神经网络的参数的梯度)
        # 计算输出层误差
        delta = self.Deriv_Cost(y_true,activation_list[-1])*af.Deriv_Sigmoid(wx_plus_b_list[-1])
        # 计算输出神经层各个参数的梯度
        nabla_biase_layer_2 = delta
        nabla_weight_layer_2 = np.dot(delta,activation_list[-2].transpose())
        # 计算中间层各个参数的梯度
        delta = np.dot(self.weights_layer_2.transpose(), delta) * \
                af.Deriv_Sigmoid(wx_plus_b_list[-2])
        nabla_biase_layer_1 = delta
        nabla_weight_layer_1 =np.dot(delta,activation_list[-3].transpose())
        return (nabla_biase_layer_1,nabla_biase_layer_2,nabla_weight_layer_1,nabla_weight_layer_2)

    """使用单个样本更新网络参数"""
    def update_in_example(self,x,y_true,learning_rate):
        # 计算每个神经层对应的参数
        nable_b_1,nable_b_2,nabla_w_1,nabla_w_2 = self.backprop(x,y_true)
        # 更新权重与偏置
        self.weights_layer_1 -=learning_rate*nabla_w_1
        self.weights_layer_2 -=learning_rate*nabla_w_2
        self.biases_layer_1 -=learning_rate*nable_b_1
        self.biases_layer_2 -=learning_rate*nable_b_2

    """使用GD算法训练神经网络"""
    def train_by_GD(self,num_epochs,data,all_y_trues,learning_rate):
        """使用训练集训练神经网络"""
        # 进行num_epochs次的训练
        for epoch in range(num_epochs):
            # 将多组样本和标签输入神经网络训练
            for x,y_true in zip(data,all_y_trues):
                self.update_in_example(x,y_true,learning_rate)
            # 定时输出训练的损失值判断效果
            if epoch %10 ==0:
                y_preds =np.apply_along_axis(self.feedforward,1,data)
                loss = MSE_Loss(all_y_trues,y_preds)
                print("Epoch %d loss: %.3f" % (epoch,loss))

    """损失函数对预测值的偏导数"""
    def Deriv_Cost(self, y_true,y_pred):
        """计算损失函数对y_pred的偏导数"""
        return (y_pred - y_true)

"""使用均方误差的损失函数,参数为数组"""
def MSE_Loss(y_true,y_pred):
    return ((y_true-y_pred)**2).mean()


if __name__=="__main__":
    # 样本
    data = np.array(
        [
            [-2,-1],
            [25,6],
            [17,4],
            [-15, -6],
        ]
    )
    # 标签
    all_y_trues = np.array(
        [
            1,
            0,
            0,
            1,
        ]
    )
    data = [x.reshape(2, 1) for x in data]
    # 创建神经网络
    network = Network()
    # 训练我们的神经网络

    network.train_by_GD(10000, data, all_y_trues,0.1)

    # 使用测试集进行测试
    emily = np.array([-7, -3])
    frank = np.array([20, 2])
    print("Emily: %.3f" % network.feedforward(emily)[0][0])
    print("Frank: %.3f" % network.feedforward(frank)[0][0])