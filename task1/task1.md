## Task 1

### 什么是Pytorch、为什么选择Pytorch？
    Pytorch是Facebook的AI团队发布的一个基于python的科学计算包，旨在服务两类场合：
    1. 替代numpy发挥GPU潜能（在线环境暂时不支持GPU）
    2. 一个提供高度灵活性和效率的深度学习实验性平台 

### Pytoch特点：
    1. PyTorch 提供了运行在 GPU/CPU 之上、基础的张量操作库; 
    2. 可以内置的神经网络库；
    3. 提供模型训练功能；
    4. 支持共享内存的多进程并发（multiprocessing ）库等；
    处于机器学习第一大语言 Python 的生态圈之中，使得开发者能使用广大的 Python 库和软件；如 NumPy、SciPy 和 Cython（为了速度把 Python 编译成 C 语言）；
    5.（最大优势）改进现有的神经网络，提供了更快速的方法——不需要从头重新构建整个网络，这是由于 PyTorch 采用了动态计算图（dynamic computational graph）结构，而不是大多数开源框架（TensorFlow、Caffe、CNTK、Theano 等）采用的静态计算图；
    6. 提供工具包，如torch 、torch.nn、torch.optim等；
### Pytorch常用工具包
    torch：类似Numpy的张量库，支持GPU；
    torch.autograd：基于tape的自动区别库，支持torch之中的所有可区分张量运行；
    torch.nn
    torch.optim：与torch.nn一起使用的优化包，包含SGD、RMSProp、LBFGS、Adam等标准优化方式；
    torch.multuprocessing:python多进程并发，进程之间torch Tensor的内存共享；
    torch.utils：数据载入器。具有训练器和其他遍历功能
    torch.legacy(..nn/..optim)：处于向后兼容性考虑，从torch一直来的legacy代码

### Pytorch安装
### 配置Python环境
### 准备python管理器
### 通过命令行安装PyToech
### PyTorch基础概念
### 通过代码实现流程：
* 数据处理阶段：深度学习依靠对大量的数据数据进行拟合，可想而知数据的处理阶段是十分重要的，所以前期我们需要花费大量的时间和精力去考虑我们所需要的数据，包括图片数据、文本数据、语音或其他的二进制数据。数据的处理对训练神经网络来说十分重要，良好的数据处理不仅会加速模型训练，更会提高模型效果。考虑到这点，pytorch提供了几个高效便捷的工具，以便使用者进行数据处理或增强等操作，同时可通过并行化加快数据的加载

* 数据加载：数据记载可通过自定的数据集对象。数据集对象被抽象为DataSet类，实现自定义的数据集需要继承Dataset，并实现两个python魔法方法: 
    1) __getitem__(self,idx)：返回对应index的数据
    2) __len__(self)：返回样本的数量。len(obj)等价于obj.__len__()

* 通过DataLoader对自定的数据集进行载入,以及对图片进行预处理
  1. resize到224
  2. crop出224
  3. 将图片数据转为tensor，以便后续计算，该函数同事进行对数值的归一化操作

* dataset下还有一个常用的ImageFolder，他的实现和上述的DogCat很相似，ImageFolder假设所有的文件按文件夹保存，每个文件夹下存储同一个类别的图片，文件夹名为类名，其构造函数如下：ImageFloder(root,transform=None, target_transform=None,loader = default_loader)

* 分类的网络：本次直接使用model中的resnet34的网络进行猫狗二分类的网络搭建，详情可以看model文件
  
* 使用损失函数：交叉熵损失函数
  
* 优化方法：直接使用Adam
  
### 注意：使用过程中发现猫狗数据集有的图片是单通道也有的图片是损坏的，所以重写dataSet，当遇到不符合要求的图片，重新在index一个



另外附上：自己实现的MTCNN，代码地址：[https://github.com/shwanliu/FaceS/tree/master/MTCNN](https://github.com/shwanliu/FaceS/tree/master/MTCNN)
