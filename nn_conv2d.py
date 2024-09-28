import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader

# 准备数据集:返回图片张量(3维)和标签的元组
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10('./data', train=False, transform=torchvision.transforms.ToTensor(), download=True)
# 加载数据集：返回图片张量(4维)和标签的元组
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        # self表示类中的全局变量，各个方法函数中都可以使用
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)      # 定义卷积对象

    def forward(self, x):
        x = self.conv1(x)       # 向对象传入参数进行使用
        return x


mymodule = MyModule()
print(mymodule)         # 显示网络结构

writer = SummaryWriter('conv2d')
step = 0
for data in dataloader:
    imgs, targets = data
    output = mymodule(imgs)
    # print(imgs.shape)             # torch.Size([64, 3, 32, 32])
    # print(output.shape)           # torch.Size([64, 6, 30, 30])
    # 注意：大于3个通道的图像无法用tensorboard来显示，则[64, 6, 30, 30]->[xxx, 3, 30, 30]，需要调整尺寸
    output = torch.reshape(output, (-1, 3, 30, 30))         # 不知道某一个值是多少时，可以用-1代替
    print(output.shape)             # [128, 3, 30, 30]
    writer.add_images('input', imgs, step)
    writer.add_images('output', output, step)
    step = step + 1

# 最大池化有时候也称为下采样
# Floor向下取整（即去掉剩余部分），ceil向上取整（也就是向上进行保留，不舍弃不完整的部分）
