import numpy as np 
import torch
import torch.nn as nn

#使用module实现lr的实现，lr是广义的线性回归，在最后一层使用了sigmoid函数，使得最后的输出为二分类，以概率输出的方式
#模型: z = wx+b ，a = sigmoid(z)
#策略: 损失函数二元交叉熵
#算法: 极大似然估计，使用梯度下降

# 随机生成数据
def createLRData(batch_size,in_channel,num_classes):
    data = torch.randn(batch_size,in_channel)
    label = torch.randint(2,(batch_size,)).long()
    return data,label

#数据归一化，避免量纲差异
def preprocess(data,type='minmax'):
    n,m = data.shape
    # 0-1标准化，min，max
    if type == 'minmax':
        # 遍历一遍一列值，即属性值，进行贵一化
        for i in range(m-1):
            minVal = torch.min(data[:,i])
            maxVal = torch.max(data[:,i])
            data[:,i] = (data[:,i] - minVal)/(maxVal-minVal)
    #Z-score，使用均值和标准差，使得分布符合标准正态分布，均值为0，标准差为1
    if type =='Z-score':
        for i in range(m-1):
            mean = torch.mean(data[:,i])
            std = torch.std(data[:,i])
            data[:,i] = (data[:,i] - mean)/std

    # sigmoid的方式
    return data

class LRmodule(nn.Module):
    def __init__(self,in_channel,out_channle):
        super(LRmodule,self).__init__()
        self.linearLayer = nn.Linear(in_channel,out_channle)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.linearLayer(x)
        x = self.sigmoid(x)
        return x


def trainLr(batch_size,in_channel,num_classes):
    # 导入模型
    lrmodel = LRmodule(in_channel,num_classes)
    # 加载数据
    data,label = createLRData(batch_size,in_channel,num_classes)
    # 数据归一化处理，
    data = preprocess(data,type='minmax')

    lossFn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lrmodel.parameters(), lr=1e-1)
    for step in range(2000):

        acc = 0
        outputs = lrmodel(data)
        # 交叉熵损失函数传的target要为一维的向量，不然会报multi-target no support
        # outs为[batch，2]
        loss = lossFn(outputs,label)
        # 使用max找到，预测的概率中最大的下标，通常都是以结果索引对应相应的类别
        _, pred = torch.max(outputs.data,1)
        acc += (pred == label).sum()
        if step%20==0:
            print("step:%d,acc:%d%%,loss:%.5f"%(step,(100 * acc/batch_size),loss))
        lrmodel.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    #生成数据
    # data,label = creatDataset(batch_size=8,in_channel=100,hidden_channel=1000,out_channle=10)
    #设定权重初始值
    # weights = initWeight([100,1000,10],2)

    # 注学习率的设置
    #train(data,label,weights,maxStep=300,lr=1e-5)

    # 我们使用numpy创建的训练集，喂入torch必须将格式进行转换
    # data = torch.from_numpy(data).float()
    # label = torch.from_numpy(label).float()

    # torch 的实现
    # trainEasyNet(data,label)


    trainLr(100,10,2)

