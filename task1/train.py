import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from DataIO import DogAndCat
from torchvision import transforms as T
from torch.utils.data import DataLoader
from model import DCNet
import datetime
import os

transform=T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])
])

def train(modelPath,dataSet,batch_size,use_cuda=True,end_epoch=10,base_lr = 0.01,frequent=20):

    trainloader= DataLoader(dataSet,batch_size=batch_size,shuffle=True,num_workers=8,pin_memory=True)
    
    if not os.path.exists(modelPath):
        os.makedirs(modelPath)

    net = sDCNet()
    if use_cuda:
        net.cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=base_lr)
    cls_loss = nn.CrossEntropyLoss()

    for cur_epoch in range(1,end_epoch):
        for batch_idx, data in enumerate(trainloader):
            image = data[0].float()
            label = data[1].long()
            # print(image.shape)
            if use_cuda:
                image=image.cuda()
                label=label.cuda()

            output = net(image)
            _,preds = torch.max(output,1)
            # print(preds)
            loss = cls_loss(output,label)
            # print(label)
            # print(pred)
            if batch_idx % frequent == 0:

                accuracy = (torch.sum(preds == label.data).float()) / batch_size
    
                loss_show = float(loss.item())

                print("%s : Epoch: %d, step:%d,accuracy:%0.5f, loss:%0.5f, lr:%s"%(
                    datetime.datetime.now(),cur_epoch,batch_idx,accuracy,loss_show, base_lr
                ))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        torch.save(net.state_dict(), os.path.join(modelPath,'epoch_%d.pt')%cur_epoch)
        torch.save(net,os.path.join(modelPath,'epoch_%d.pkl')%cur_epoch,)

if __name__ == "__main__":
    data = DogAndCat('../dataSet/PetImages/',transform=transform)
    train(modelPath="./store",dataSet=data,batch_size=8,use_cuda=True,end_epoch=10,base_lr=0.01,frequent=20)