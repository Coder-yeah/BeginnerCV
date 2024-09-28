from torch.utils.data import Dataset  # 常用工具区
from PIL import Image  # 读取图片
import os
# Dateset工具主要用来整合数据，例如记录图像数据的类别、编号、数量以及位置等


class MyData(Dataset):  # Dataset是一个抽象类:所有数据集都需要继承这个类，相当于一个模板，我们需要重写三个方法
    def __init__(self, root_dir, label_dir):  # 初始化，创建一个特例实例时需要的函数，记录class的全局变量，为后面的方法提供所需要的量
        self.root_dir = root_dir  # self用于指定一个类中的全局变量，可以跨函数使用
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):  # 获取相应的图片,需要先有图片列表所需的变量（在init中定义）
        img_name = self.img_path[idx]  # self表示引用的是全局变量
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label           # 返回图片的PIL对象和标签对

    def __len__(self):
        return len(self.img_path)


root_dir = 'datasets'
sansha_label_dir = '小目标'
yunyan_label_dir = '云烟'
sansha_dataset = MyData(root_dir, sansha_label_dir)       # 创建实例
yunyan_dataset = MyData(root_dir, yunyan_label_dir)

train_dataset = sansha_dataset + yunyan_dataset
print(train_dataset[0])     # 直接向对象传入索引使用__getitem__方法返回某一张PIL图片以及类别
# 返回(<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=313x495 at 0x26772FA6198>, '小目标')
