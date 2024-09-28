import torchvision

# 准备测试数据集
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_set = torchvision.datasets.CIFAR10('./data', train=False, transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

# 测试数据集中的第一张图片及target，loader时按照sampler的方式（默认是随机取）开始取出并分别对图片和标签进行打包
img, target = test_set[0]
print(img.shape)
print(target)

# num_workers是否采用多进程加载数据，在windows下>0会报错，出现BrokenPipeError
# drop_last参数决定余数部分的图片是否进行打包加载
# shuffle（洗牌）参数使得每一次（每一个epoch）读取到的图片batch都有所不同
writer = SummaryWriter('dataloader')
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data        # 每次抓取之后返回图片与标签的打包表示
        # print(imgs.shape)           # 返回一个4维tensor表示的图片
        # print(targets)                #返回一个标签列表
        writer.add_images('epoch:{}'.format(epoch), imgs, step)       # 多于3通道的图片需要使用add_images
        step = step + 1

writer.close()
