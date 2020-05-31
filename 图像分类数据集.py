import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
sys.path.append('..')
import d2lzh_pytorch as d2l

mnist_train = torchvision.datasets.FashionMNIST(root='C:\code\python\pytorch\动手深度学习', train=True, download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='C:\code\python\pytorch\动手深度学习', train=False, download=True, transform=transforms.ToTensor())
print(type(mnist_train), len(mnist_train))
print(type(mnist_test), len(mnist_test))

feature, label = mnist_train[0]
print(feature.shape, label)


def get_fashion_mnist_label(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag',
                   'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(images, labels):
    d2l.use_svg_display()
    _, figs = plt.subplots(1, len(images), figsize=(28, 28))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_xaxis().set_visible(False)
    plt.show()


x, y = [], []
for i in range(10):
    x.append(mnist_train[i][0])
    y.append(mnist_train[i][1])
show_fashion_mnist(x, get_fashion_mnist_label(y))

batch_size = 256
if sys.platform.startswith('win'):
    num_workers = 0
else:
    num_workers = 4
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=num_workers)
train_iter = torch.utils.data.DataLoader(mnist_test, batch_size, shuffle=True, num_workers=num_workers)

start = time.time()
for x, y in train_iter:
    continue
print('%.2f' % (time.time()-start))


