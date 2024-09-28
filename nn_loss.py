import torch
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss

# 计算时只能用浮点数，不能用Long整型
inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)

inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

loss = L1Loss()         # 创建对象
result = loss(inputs, targets)          # 使用方法
print(result)

loss_mse = MSELoss()
result_mse = loss_mse(inputs, targets)
print(result_mse)

# 注意：输入形状（N, C)--> ，目标形状（N）--> batch中图片的标签值
x = torch.tensor([0.1, 0.2, 0.3])           # 预测得出的各个类别的概率
y = torch.tensor([1])           # 目标是通过概率得出类别1
x = torch.reshape(x, [1, 3])           # batchsize=1, class=3
print(x)
cross_loss = CrossEntropyLoss()
result_cross = cross_loss(x, y)         # 概率和类别都正确时，loss最小 -> 由公式中的-x[class]
print(result_cross)
