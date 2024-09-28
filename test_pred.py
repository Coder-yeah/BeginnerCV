import torch

# 举例：二分类问题
# 模型输出
outputs = torch.tensor([[0.1, 0.2],
                       [0.3, 0.4]])
# argmax方法可以将概率最大的值转换为其所在位置（标签值），1是横向看，0是纵向看
print(outputs.argmax(1))            # tensor([1, 1])
preds = outputs.argmax(1)           # 输出标签
targets = torch.tensor([0, 1])      # 真实标签
print(preds == targets)             # tensor([False,  True])，True=1, False=0，
print((preds == targets).sum())       # tensor(1)，转换为数字，即True的个数
