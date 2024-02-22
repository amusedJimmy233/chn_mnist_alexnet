# -*- coding: utf-8 -*-
# @Time    : 2022/12/21 16:30
# @Author  : Jimmy
# @Email   : 2362763425@qq.com
# @File    : train.py

import torch
from torch import optim
from tqdm import tqdm
from matplotlib import pyplot as plt


def train(net, device, lr, momentum, epochs, train_dataloader):
    # 损失函数:这里用交叉熵
    criterion = torch.nn.CrossEntropyLoss()
    # 优化器 这里用SGD
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    if torch.cuda.is_available():
        print("device:GPU CUDA")
    else:
        print("device:CPU")

    print("Start Training!")

    x_data = []
    y_data = []
    total_loss = 0

    for epoch in range(epochs):

        # len(train_dataloader)=len(train_dataset)/batch_size
        loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader))

        for i, data in loop:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            labels = labels.to(device, dtype=torch.long)  # 这里需要转换dtype类型，不然会报错
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            # 将梯度清零
            optimizer.zero_grad()
            # 对损失函数进行反向传播
            loss.backward()
            # 训练
            optimizer.step()

            # 更新信息
            loop.set_description(f'Epoch [{epoch + 1}/{epochs}]')
            loop.set_postfix(loss=loss.item())

        # 计算每一轮的平均loss
        avg_loss = total_loss / len(train_dataloader)
        y_data.append(avg_loss)
        total_loss = 0
        x_data.append(epoch + 1)

    print("Finished Traning")

    # 保存训练模型
    torch.save(net, 'MNIST.pkl')

    # 可视化
    plt.plot(x_data, y_data)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()
