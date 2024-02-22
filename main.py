# -*- coding: utf-8 -*-
# @Time    : 2022/12/21 16:30
# @Author  : Jimmy
# @Email   : 2362763425@qq.com
# @File    : main.py


from net import AlexNet
from load_data import load_data
from train import train
from test import test
import torch

# 超参数设置
EPOCH = 10  # 遍历数据集次数
BATCH_SIZE = 128  # 批处理尺寸
LR = 0.001  # 学习率
MOMENTUM = 0.9  # 动量
TEST_SIZE = 0.3  # 测试集在总数据集中的比例

if __name__ == '__main__':
    # 定义是否使用GPU训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建网络
    net = AlexNet()
    # 将训练硬件加载进神经网络
    net.to(device)
    # 加载数据集，并转化为能够直接使用的dataloader形式
    train_dataloader, test_dataloader = load_data(BATCH_SIZE, TEST_SIZE)
    # 进行训练，将训练参数传入
    train(net, device, LR, MOMENTUM, EPOCH, train_dataloader)
    # 进行测试，并打印结果
    test(device, test_dataloader)
