import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models

class DCNet(nn.Module):
    def __init__(self):
        super(DCNet,self).__init__()
        model = models.resnet34(pretrained = True)
        self.resnet_layer = nn.net = nn.Sequential(*list(model.children())[:-2])

        self.conv1_layer = nn.Conv2d(512,256,1,1)
        self.relu1_layer = nn.PReLU()
        self.drop_layer = nn.Dropout2d(0.5)
        self.global_average = nn.AdaptiveAvgPool2d((1,1))
        self.fc_linear = nn.Linear(256,2)
    
    def forward(self,x):
        x = self.resnet_layer(x)
        x = self.conv1_layer(x)
        x = self.relu1_layer(x)
        x = self.drop_layer(x)
        x = self.global_average(x)
        x = x.view(x.size(0),-1)
        x = self.fc_linear(x)
        return x

if __name__ == "__main__":
    input_ = torch.randn(1,3,224,224)
    net = DCNet()
    x=net(input_)
    print(DCNet())
    print(x)
