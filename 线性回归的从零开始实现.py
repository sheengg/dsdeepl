import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import random

#  定义需要的数据
num_iputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.randn(num_examples, num_iputs, dtype=torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b  # 计算标签
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)  # 标签添加噪声


# 读取数据
def data_iter(batch_size, features, lables):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])
        yield features.index_select(0, j), labels.index_select(0, j)


batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, y)
    break

w = torch.tensor(np.random.normal(0, 0.01, (num_iputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)

w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)


#  定义模型
def linreg(X, w, b):
    return torch.mm(X, w) + b


#  定义损失函数
def squared_loss(y_hat, y):
    return (y_hat-y.view(y_hat.size())) ** 2 / 2


#  定义优化方法
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size


#  训练模型
lr = 0.03
num_epoch = 3  # 需要迭代的次数
net = linreg
loss = squared_loss

for epoch in range(num_epoch):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()
        l.backward()
        sgd([w, b], lr, batch_size)

        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)
    print("epoch %d, loss %f" % (epoch + 1, train_l.mean().item()))
