# -*- coding: utf-8 -*-
# @Time    : 2022/12/24 22:33
# @Author  : Jimmy
# @Email   : 2362763425@qq.com
# @File    : test.py

import torch


def test(device, test_dataloader):
    net = torch.load('MNIST.pkl')
    # 开始识别
    with torch.no_grad():  # 在接下来的代码中，所有Tensor的requires_grad都会被设置为False，不会计算梯度
        correct = 0
        total = 0

        for data in test_dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            out = net(images)
            _, predicted = torch.max(out.data, 1)
            total += labels.size(0)  # labels.size(0)即batch_size,每次处理的数量
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the {} test images:{}%'.format(total, 100 * correct / total))  # 输出识别准确率
