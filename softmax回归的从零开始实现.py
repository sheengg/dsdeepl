import torch
import torchvision
import numpy as np
import d2lzh_pytorch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10

w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
b = torch.zeros(num_outputs, dtype=torch.float)

w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)


#  定义 softmax 函数
def softmax(x):
    x_exp = x.exp()
    partition = x_exp.sum(dim=1, keepdim=True)
    return x_exp / partition


# 定义网络
def net(x):
    return softmax(torch.mm(x.view((-1, num_inputs)), w) + b)


#  定义交叉熵
def corss_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))


#  计算分类准确度
def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()


#  计算预测准确度
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for x, y in data_iter:
        acc_sum += (net(x).argmax(dim=1)==y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


#  训练模型
num_epochs, lr = 5, 0.1


def train_ch3(net, train_iter, loss, num_epochs, batch_size, params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for x, y in train_iter:
            y_hat = net(x)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            l.backward()
            if optimizer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                optimizer.step()

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1)==y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print("epoch %d, loss %f, train acc %f, test acc %f" % (epoch+1, train_l_sum / n, train_acc_sum/n, test_acc))


train_ch3(net, train_iter, corss_entropy, num_epochs, batch_size, [w, b], lr)


#  测试

x, y = iter(test_iter).next()

true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(x).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

d2l.show_fashion_mnist(x[0:9], titles[0:9])





