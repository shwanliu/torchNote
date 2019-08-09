import torch
import torch.utils.data as data
import cv2
import numpy
from PIL import Image
import os
from torchvision import transforms as T
import random
transform=T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    # T.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])
])
# 读取数据的base类，
class baseDataset(data.Dataset):
    def __init__(self,dirs,transform=None):
        # 获得该目录下每一张图片的绝对路径
        imageDir = os.listdir(dirs)
        self.image = []
        imagePaths= [os.path.join(dirs,image) for image in imageDir]
        self.transform = transform

        # self.image = [os.path.join(dirs,image) for image in images]
        for imagePath in imagePaths:
            self.image+=[os.path.join(imagePath,image) for image in os.listdir(imagePath)]
        # print(self.image)
    def __getitem__(self,index):
        data = self.image[index] 
        label = 1 if 'dog' in self.image else 0
        data = Image.open(data).convert('RGB')
        if self.transform:
            data = self.transform(data)
        return data,label
    def __len__(self):
        return len(self.image)

#保证batch_size里面的数据都是有效的，有的图片是坏的
class DogAndCat(baseDataset):
    def __getitem__(self,index):
        try:
            return super(DogAndCat,self).__getitem__(index)
        except:
            print("bad image")
            new_index = random.randint(0,len(self)-1)
            return self[new_index]

if __name__ == "__main__":
    d = DogAndCat('../dataSet/PetImages/',transform=transform)
    print(d[10])
    print(len(d))