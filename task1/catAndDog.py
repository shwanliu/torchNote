import torch
import torch.utils as data
import cv2
import os

class DogAndCat(data.Dataset):
    def __init__(self,dirs):
        # 获得该目录下每一张图片的绝对路径
        images = os.listfile(dirs)
        # 
        self.image = [os.listdir(os.path.join(dirs,image)) for image in images]


    def __getitem__(self,index):
        image = self.image[index] 
        label = 1 if 'dog' in self.image else 0
        return image,label
    def __len__(self):
        return len(self.image)