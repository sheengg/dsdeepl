import torch
import numpy as np
import torch.utils.data as Data
from torch.nn import init
import torch.nn as nn
import torch.optim as optim
#  定义需要的数据
num_iputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.randn(num_examples, num_iputs, dtype=torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b  # 计算标签
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)  # 标签添加噪声

batch_size = 10
dataset = Data.TensorDataset(features, labels)  # 将训练数据的特征和标签组合
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)  # 随机小批量读取数据
#  data_iter 和的使用和上一节的一样，打印一下看看
for X, y in data_iter:
    print(X, y)
    break


#  定义模型
class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    def forward(self, x):
        y = self.linear(x)
        return y


net = LinearNet(num_iputs)
print(net)
net = nn.Sequential(
    nn.Linear(num_iputs, 1)
    # 其他层

)
#  初始化模型参数
init.normal_(net[0].weight, mean=0, std=0.01)
init.constant_(net[0].bias, val=0)
#  定义损失函数
loss = nn.MSELoss()
#  定义优化方法
optimizer = optim.SGD(net.parameters(), lr=0.003)
#  训练模型
num_epoch = 30
for epoch in range(1,num_epoch + 1):
    for x, y in data_iter:
        output = net(x)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))

