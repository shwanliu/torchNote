import numpy as np 
import torch
import torch.nn as nn

#使用numpy 设计一个网络，仅有一层隐藏层
def creatDataset(batch_size,in_channel,hidden_channel,out_channle):
    x = np.random.randn(batch_size,in_channel)
    y = np.random.randn(batch_size,out_channle)
    return x,y

def initWeight(channel,hidden_layers):
    weights=[]
    for i in range(1,hidden_layers):
        weight=[]
        weights.append(np.random.randn(channel[i-1],channel[i]))
        weight=[]
        weights.append(np.random.randn(channel[i],channel[i+1]))
    return weights

def train(x,y,weights,maxStep,lr):

    for step in range(maxStep):
        
        h = x.dot(weights[0])
        # print(h)
        h_relu = np.maximum(h,0)
        pred = h_relu.dot(weights[1])
        # print(pred)
        loss = np.square(pred - y).sum()
        print("step:%d,loss:%.5f"%(step,loss))

        #这里计算梯度进行反向传播
        # backward 计算梯度
        grad_pred= 2.0*(pred -y)
        # print(grad_pred)
        grad_w2 = h_relu.T.dot(grad_pred)
        grad_h_relu = grad_pred.dot(weights[1].T)
        grad_h =  grad_h_relu.copy()
        grad_h[h<0] = 0
        grad_w1 = x.T.dot(grad_h)

        # 在梯度方向上进行参数的更新，根据梯度进行更新
        weights[0] -= lr* grad_w1 
        weights[1] -= lr* grad_w2


#使用torch实现梯度下降
class easyNet(nn.Module):
    def __init__(self):
        super(easyNet,self).__init__()
        
        self.linear1 = nn.Linear(100,1000,bias=False)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(1000,10,bias=False)

    def forward(self,x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

if __name__ == "__main__":
    #生成数据
    data,label = creatDataset(batch_size=8,in_channel=100,hidden_channel=1000,out_channle=10)
    #设定权重初始值
    weights = initWeight([100,1000,10],2)
    # 注学习率的设置
    #train(data,label,weights,maxStep=300,lr=1e-5)
    # 我们使用numpy创建的训练集，喂入torch必须将格式进行转换
    data = torch.from_numpy(data).float()
    label = torch.from_numpy(label).float()
    # torch 的
    net = easyNet()
    print(net)
    print(data.shape)
    mseLoss = nn.MSELoss()
    for step in range(100):
        output = net(data)
        loss =mseLoss(output,label)
        print("step:%d,loss:%.5f"%(step,loss))

        net.zero_grad()
        # 直接反向传播
        loss.backward()

        #这边不直接使用优化类optimizer来进行参数更新
        with torch.no_grad():
            for param in net.parameters():
                param.data -= (1e-5)*param.grad
        #直接使用optimizer更新参数
        # optim = optimizer = torch.optim.Adadelta(net.parameters(), lr=1e-5)
        # optim.step()



