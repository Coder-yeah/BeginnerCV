import torch
import torchvision.datasets
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, -0.5],
                      [-1, 3]])
input = torch.reshape(input, (-1, 1, 2, 2))         # relu的输入也是一个四维张量
print(input)

dataset = torchvision.datasets.CIFAR10('./data', train=False, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)


# 非线性变换的目的：对图像引入非线性处理，变换图像的明暗关系，让网络拟合各种曲线，提高模型的泛化能力
class Mymodule(nn.Module):
    def __init__(self):
        super(Mymodule, self).__init__()
        self.relu1 = ReLU()         # inplace参数：是否对原来变量的结果进行变换，是否需要丢失原始数据
        self.sigmoid1 = Sigmoid()

    def forward(self, x):
        output = self.relu1(x)
        return output


mymodule = Mymodule()
# output = mymodule(input)
# print(output)

writer = SummaryWriter('relu')
step = 0
for data in dataloader:
    imgs, targets = data
    output = mymodule(imgs)
    writer.add_images('imgs', imgs, step)
    writer.add_images('relu', output, step)
    step = step+1

writer.close()

# 正则化成的参数1：对应输入图片的信道数
# Recurrent层：主要是文字识别的网络结构
# transform层：transformer
# 线形层：常用，搭建全连接的线性网络，三个参数：输入神经元数(通道数)、输出神经元数（通道数）、偏置
# dropout层：随机把张量中的某个元素置为0，防止过拟合
# sparse层：主要用于自然语言处理层

