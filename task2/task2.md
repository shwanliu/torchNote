# Task2 

### numpy和pytorch实现梯度下降法

* 设定初始值 \
   在神经网络中的权重初始化犯法对模型的手乱速度和性能有着至关重要的影响，说白了，神经网络其实就是对权重参数的不断迭代更新，以期望达到比较好的性能，在深度神经网络中，随着层数的不断增加，在梯度下降的过程中，十分容易出现梯度消失一节梯度爆炸的情况，因此，定义权重的出事就显得十分重要，一般有以下集中权重初始化方法：
   1. 将w 初始化为0: \
   我们在做线性回归、逻辑斯蒂回归的时候，基本上都是把参数初始化为0，我们的的模型也能很好的工作。然后在神经网络中，将w初始化为0是不可以的，这是因为如果把w初始化为0，那么每一层的神经元学到的东西都是一样的，而且在bp的时候，每一层内的神经元也是相同的， 
   >parameters['W' + str(l)] = np.zeros((layers_dims[l], layers_dims[l - 1])) \
   >parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
   
   2. 对w随机初始化：这是目前比较常用的随机初始化方法，但是由于随机初始化其实是对一个均值为0，方差为1的高斯分布进行采样，当网络层数增多，越后面的输出值几乎接近于0
    > parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1])*0.01 \
    > parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
   
   3. Xavier 初始化：Xavier initialization是 Glorot 等人为了解决随机初始化的问题提出来的另一种初始化方法，他们的思想倒也简单，就是尽可能的让输入和输出服从相同的分布，这样就能够避免后面层的激活函数的输出值趋向于0。他们的初始化方法为：
    >parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(1 / layers_dims[l - 1])  \
    >parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

   4. He 初始化
    >parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(2 / layers_dims[l - 1]) \
    >parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

* 求取梯度
  1. 根据网络的最后一层开始计算差值，使用到链式求导法则

* 在梯度方向上进行参数的更新
  根据:
    $$ \theta_t = \theta_{t-1} - \alpha $$ 
    
### numpy和pytorch实现线性回归
实现曲线拟合，其中的使用的特征越多，即权重不为0的项，拟合效果好，但是过拟合了，

### pytorch实现一个简单的神经网络
实现一个仅有一个隐藏层的网络 输入100个节点，隐藏层1000个节点，输出10个节点，在代码中的easyNet

### 参考资料：PyTorch 中文文档     https://pytorch.apachecn.org/docs/1.0/