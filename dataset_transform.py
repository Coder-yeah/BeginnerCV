import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, transform=dataset_transform, download=True)

"""
print(test_set[0])          # 与前面的操作类似，直接传入索引获取数据集中的某一个PIL图片和类别
print(test_set.classes)
img, target = test_set[0]
print(img, target)
print(test_set.classes[target])
img.show()      # PIL对象可以直接使用PIL的方法
"""

# 加入transform，将图片转换成tensor类型，使用tensorboard显示
# 注意test_set[0]是一个包含图片和标签类别的元组
# print(test_set[0])

writer = SummaryWriter('CIFAR_logs')
for i in range(10):
    img, target = test_set[i]
    writer.add_image('test', img, i)
writer.close()
