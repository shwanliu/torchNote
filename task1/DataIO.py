import torch
import torch.utils.data as data
import cv2
import os

class DogAndCat(data.Dataset):
    def __init__(self,dirs):
        # 获得该目录下每一张图片的绝对路径
        imageDir = os.listdir(dirs)
        print(imageDir)
        self.image = []
        imagePaths= [os.path.join(dirs,image) for image in imageDir]
        print(imagePaths)
        # 
        # self.image = [os.path.join(dirs,image) for image in images]
        for imagePath in imagePaths:
            self.image+=[os.path.join(imagePath,image) for image in os.listdir(imagePath)]
        # print(self.image)
    def __getitem__(self,index):
        image = self.image[index] 
        label = 1 if 'dog' in self.image else 0
        return image,label
    def __len__(self):
        return len(self.image)

if __name__ == "__main__":
    d = DogAndCat('../dataSet/PetImages/')
    print(d[10])
    print(len(d))