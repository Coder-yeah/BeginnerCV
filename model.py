import torch
from torch import nn


class module(nn.Module):
    def __init__(self):
        super(module, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


# 可以在main程序中验证网络的正确性：输入、输出的尺寸变化是否正确
if __name__ == '__main__':
    input = torch.ones((64, 3, 32, 32))
    m = module()
    output = m(input)
    print(output.shape)
