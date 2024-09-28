import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10('./data', train=False, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)


class Mymodule(nn.Module):
    def __init__(self):
        super(Mymodule, self).__init__()
        self.linear1 = Linear(196608, 10)           # 注意：定义实例时需要传参

    def forward(self, x):
        x = self.linear1(x)             # 1x1x196608 -> 1x1x10
        return x


m = Mymodule()
for data in dataloader:
    imgs, targets = data
    print(imgs.shape)           # torch.Size([64, 3, 32, 32])
    # input = torch.reshape(imgs, (1, 1, 1, -1))
    # print(input.shape)          # torch.Size([1, 1, 1, 196608])
    input = torch.flatten(imgs)
    print(input.shape)          # torch.Size([196608])
    output = m(input)
    print(output.shape)         # torch.Size([1, 1, 1, 10]) -> torch.Size([10])
