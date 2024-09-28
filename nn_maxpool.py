import torch
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader


# 池化的作用：保留特征，但是可以减少数据量，网络的训练参数减少
# 准备数据集
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10('./data', train=False, transform=torchvision.transforms.ToTensor(), download=True)
# 加载数据集
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
# 注意：需要转化成浮点数进行最大池化处理，常用的数据处理类型的也应该为浮点数
input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]], dtype=torch.float32)
input = torch.reshape(input, (-1, 1, 5, 5))      # 池化操作的输入尺寸也是四维
print(input.shape)          # [1, 1, 5, 5]


class Mymodule(nn.Module):
    def __init__(self):
        super(Mymodule, self).__init__()
        self.maxpool_1 = MaxPool2d(3, ceil_mode=True)

    def forward(self, x):
        x = self.maxpool_1(x)
        return x


mymodule = Mymodule()
# output = mymodule(input)
# print(output)

writer = SummaryWriter('maxpool')
step = 0
for data in dataloader:
    imgs, targets = data
    output = mymodule(imgs)
    print(output)
    writer.add_images('img', imgs, step)
    writer.add_images('output', output, step)
    step = step + 1
writer.close()

# 非线性激活