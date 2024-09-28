import torchvision.datasets
from torch.nn import Linear
# train_data = torchvision.datasets.ImageNet('./img', split='train', download=True,
#                                            transform=torchvision.transforms.ToTensor())

# 加载现有网络模型
vgg16_False = torchvision.models.vgg16(pretrained=False)
vgg16_True = torchvision.models.vgg16(pretrained=True)

print(vgg16_True)

dataset = torchvision.datasets.CIFAR10('./data', train=False, transform=torchvision.transforms.ToTensor())

# 修改现有网络架构
# 1、添加层：例如在vgg16后面添加一层
vgg16_True.add_module('add_linear', Linear(1000, 10))
# 如果需要加入某一层中（例如classifier层），则
vgg16_True.classifier.add_module('7', Linear(1000, 10))
# print(vgg16_True)
# 2、修改层
vgg16_False.classifier[6] = Linear(4096, 10)
print(vgg16_False)
