"""
2025.11.01
4.2 多层感知机的从零开始实现
此部分代码完全可以实现，完全手敲理解的
"""
import torch
from torch import nn
# from d2l import torch as d2l
from torch.utils import data
from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt


# train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# 自定义数据加载，可以自己设置下载路径
"""
trans = [transforms.ToTensor()]
if resize:
trans.insert(0, transforms.Resize(resize))，
在计算机视觉中，数据预处理通常遵循这个顺序：原始图像 → 调整尺寸 → 转换为张量 → 归一化，
transforms.ToTensor() 不仅将图像转换为张量，还会：自动将像素值从 [0, 255] 缩放到 [0, 1]，改变维度顺序：从 (H, W, C) 变为 (C, H, W)
错误的顺序：先转张量再调整尺寸，trans = [transforms.ToTensor(), transforms.Resize(64)]  # 错误！
问题：ToTensor() 之后得到的是 (C, H, W) 的张量
但 transforms.Resize() 期望输入是 (H, W, C) 的PIL图像
正确的顺序：先调整尺寸再转张量
trans = [transforms.ToTensor()]  # 基础转换
if resize:
    trans.insert(0, transforms.Resize(resize))  # 在开头插入
# 最终顺序： [Resize(resize), ToTensor()]
"""
"""
transforms.Compose(): PyTorch中的一个类，用于将多个转换操作串联起来
trans: 之前定义的转换操作列表
返回一个可执行的转换管道，可以按顺序对图像应用所有转换
"""
def load_data_fashion_mnist(batch_size,resize=None):
    trans = [transforms.ToTensor()]
    print("type(trans) = ",type(trans))
    if resize:
        trans.insert(0,transforms.Resize(resize))
    # 前面几行是设置图像操作顺序
    # 下面一行进行图像处理
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root="./data",train=True,transform=trans,download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="./data",train=False,transform=trans,download=True)
    return data.DataLoader(mnist_train, batch_size, shuffle=True), data.DataLoader(mnist_test, batch_size, shuffle=False)
batch_size = 64
train_iter, test_iter = load_data_fashion_mnist(batch_size)
# 4.2.1 初始化模型参数

"""
使用 nn.Parameter 的最主要原因是：
告诉 nn.Module 这个张量是模型的可训练参数（learnable parameters）
nn.Parameter 表面上看起来只是对 Tensor 做了一个简单的包装，但它在 PyTorch 的 nn.Module 框架下有一个关键作用：
自动注册： 将一个张量标记为模块（nn.Module）的参数。
自动追踪： 使得 model.parameters() 方法能够自动找到它。
启用优化： 只有 model.parameters() 列表中的张量才会被传递给优化器（如 SGD, Adam）进行训练和更新。
补充说明： 正如您代码中所示，nn.Parameter 包装的张量默认 requires_grad=True。 所以，你写的 requires_grad=True 实际上是多余的（但无害）。更简洁的写法是：
W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens) * 0.01)
PyTorch会自动将其 requires_grad 属性设为 True。
"""
num_inputs, num_outputs, num_hiddens = 784, 10, 256
W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens,requires_grad=True))
W2 = nn.Parameter(torch.randn(num_hiddens,num_outputs,requires_grad=True)*0.01)
b2 = nn.Parameter(torch.zeros(num_outputs,requires_grad=True))
params = [W1, b1, W2, b2]

# 4.2.2 激活函数
def relu(x):
    a = torch.zeros_like(x)
    return torch.max(x,a)

# 4.2.3 模型
"""
为什么要用两个圆括号 ( ) ？
这是一个关于 Python 函数语法的常见疑问。

最外层的括号 x.reshape(...)： 这是调用 reshape 这个**方法（函数）**时必须写的括号。

最内层的括号 (-1, num_inputs)： 这是在创建一个 元组（Tuple）。

reshape 函数（无论是 PyTorch 还是 NumPy）在技术上只接受 一个 参数来指定新的形状。这个参数必须是一个代表形状的元组（或者列表）。

所以，((-1, num_inputs)) 的意思是： “我正在调用 reshape 函数，我传给它的那一个参数是元组 (-1, num_inputs)。”

补充： 很多人会省略内层的括号，写成 x.reshape(-1, num_inputs)。

这种写法在 PyTorch 和 NumPy 中也是允许的。这是一种“语法糖”（syntactic sugar），库的设计者允许你把元组的各个元素直接当作单独的参数传入，
库内部会自动帮你把它们组装成一个元组。

但最严格、最明确的写法是 x.reshape((-1, num_inputs))，它清楚地表明你正在传入一个元组。两种写法效果完全一样。
"""
"""
-1 和 num_inputs 是什么意思？这行代码的目的是把任意形状的输入 x “压平”成一个二维矩阵，
以便进行后续的矩阵乘法（x @ W1）。这个二维矩阵的形状被指定为 (行, 列)，也就是 (-1, num_inputs)。
num_inputs (列数)这是一个变量（在代码的其他地方定义），它代表**“每个数据样本有多少个特征”**。
例如，如果你的输入数据是 28x28 像素的黑白图片，那么 num_inputs 变量的值就应该是 $28 * 28 = 784$。
这行代码强制指定了新矩阵的列数必须是 num_inputs。-1 (行数)这是 reshape 功能中的一个特殊占位符。
它告诉 PyTorch/NumPy：“我懒得算这一维（行数）应该是多少，请你根据原始数据的总元素量和固定的列数（num_inputs）自动帮我算出来。
”在神经网络的上下文中，这个自动算出来的行数就是批量大小（Batch Size），即你这一次喂进来了多少个数据样本。
"""
"""
x = x.reshape((-1, num_inputs))
重塑（Flattening）输入。这是一个预处理步骤。
num_inputs 是一个（在代码别处定义的）变量，代表每个样本的特征数量（例如，一个28x28的图像，num_inputs 就是 784）。
-1 是一个占位符，意思是“自动计算这一维的大小”。它代表批量大小（batch size），即你一次性喂给网络多少个样本。
作用： 无论输入的 x 原本是什么形状（比如 (64, 1, 28, 28)），这行代码都会把它“压平”成一个二维矩阵，形状为 (批量大小, 特征数量)。这是为了让它能和权重矩阵 W1 进行矩阵乘法。
"""
def net(x):
    x = x.reshape((-1,num_inputs))
    # 隐藏层的输出确实进行了ReLU激活。
    H = relu(x@W1+b1)
    return (H@W2+b2)

# 4.2.4 损失函数
"""
reduction='none' 的意思是：不对批量（batch）中每个样本的损失（loss）进行聚合。
简单来说，它会返回一个包含“每个样本各自损失”的张量（tensor），而不是返回一个单一的平均值或总和。

reduction 参数决定了如何处理这 N个损失值：
reduction='mean' (默认选项)含义： 取平均值。
reduction='sum' 含义： 求和。
reduction='none' 含义： 不聚合，返回全部。
"""
loss = nn.CrossEntropyLoss(reduction='none')
num_epochs, lr = 10, 0.15
updater = torch.optim.SGD(params,lr=lr)
# updater.step()
# print("updater.param_groups = ", updater.param_groups)

"""
if isinstance(updater, torch.optim.Optimizer): 
检查变量 updater 是否是 PyTorch 优化器对象。

l.mean().backward() 是PyTorch中计算梯度的关键操作

updater.step() 是深度学习训练中的核心操作，它：
根据计算出的梯度更新模型参数
实现了优化算法的具体更新规则
是模型从数据中学习的关键步骤
必须与 zero_grad() 和 backward() 配合使用
"""
"""
自定义 updater.step() 的规则：
方法1：自定义优化器类
import torch
from torch.optim import Optimizer

class CustomSGD(Optimizer):
    def __init__(self, params, lr=0.01, weight_decay=0):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(CustomSGD, self).__init__(params, defaults)
    
    def step(self, closure=None):
        # 执行自定义的参数更新规则
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                # 自定义更新规则：添加动量-like 效果
                if hasattr(p, 'momentum_buffer'):
                    buf = p.momentum_buffer
                    grad = 0.9 * buf + 0.1 * grad
                    p.momentum_buffer = grad.clone()
                else:
                    p.momentum_buffer = grad.clone()
                
                # 应用权重衰减
                if weight_decay != 0:
                    grad = grad + weight_decay * p.data
                
                # 更新参数
                p.data.add_(-lr, grad)
        
        return loss

# 使用自定义优化器
model = torch.nn.Linear(10, 1)
updater = CustomSGD(model.parameters(), lr=0.01)

方法3：修改现有优化器
class ModifiedSGD(torch.optim.SGD):
    def step(self, closure=None):
        # 修改现有的SGD优化器
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # 在标准SGD之前添加自定义逻辑
                grad = p.grad.data
                
                # 示例：对梯度进行特殊处理
                if p.data.norm() > 1.0:  # 如果参数范数太大
                    grad = grad * 0.5     # 减小梯度
                
                # 调用父类的更新逻辑
                p.data.add_(-group['lr'], grad)
        
        return loss
"""
"""
单个batch训练函数
else部分：
l.sum().backward(): 保持原始梯度尺度，不进行平均
updater(X.shape[0]): 自定义优化器函数，可以接收批量大小等参数
updater(X.shape[0])为什么要使用X.shape[0]？
回答：为了获取当前批次的样本数量。
那为什么不直接用l.mean().backward()？
回答：数学上，这两种方法是等价的。
l.mean().backward(): PyTorch自动从张量形状推断批次大小，梯度自动归一化
updater(X.shape[0]): 需要手动获取并传递批次大小，手动归一化，updater(X.shape[0]) 这种方法基本上用不到
"""

"""
评估和累加函数：
metric = Accumulator(2)
创建 Accumulator 实例，初始化2个计数器
metric.data 初始化为 [0.0, 0.0]
第一个位置存储：正确预测的总数
第二个位置存储：总样本数。

metric.add(accuracy(net(X), y), y.numel())
accuracy(net(X), y)：计算当前批次的正确预测数量
y.numel()：计算当前批次的样本数量（y中元素的总数）
metric.add(a, b)：将两个值分别加到累加器的两个计数器

def add(self, *args):
    self.data = [a + float(b) for a, b in zip(self.data, args)]
*args：接受任意数量的参数
zip(self.data, args)：将累加器的当前值与新值配对
[a + float(b) for ...]：将每对值相加，更新累加器

[a + float(b) for ...]：将每对值相加，更新累加器，是如何更新的？不太理解
答：例如，执行：metric.add(8, 10)，其中args = (8, 10)，self.data = [0.0, 0.0]
那么，zip(self.data, args) 
# → zip([0.0, 0.0], (8, 10))
# → 生成: [(0.0, 8), (0.0, 10)]，
然后，[a + float(b) for a, b in [(0.0, 8), (0.0, 10)]]
# 第一个元素: 0.0 + float(8) = 0.0 + 8.0 = 8.0
# 第二个元素: 0.0 + float(10) = 0.0 + 10.0 = 10.0
# → 结果: [8.0, 10.0]
则，self.data = [8.0, 10.0]  # 替换原来的 [0.0, 0.0]
"""
# 自己写训练函数
print("type(torch.optim) = ", type(torch.optim))
"""
训练步骤：
1.设置训练模式
2.计算预测值、损失
3.批量数据的梯度置零、反向传播计算梯度值、根据梯度值更新模型参数
4.计算损失、平均精度
    计算损失：自动计算
    精度计算：预测正确个数 / 总预测个数
"""
def accuracy(y_hat, y): # 返回预测正确的个数
    """
    y_hat将是（批量数*特征数量
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    条件1：len(y_hat.shape) > 1
    检查 y_hat 是否是多维张量（不是一维向量）
    条件2：y_hat.shape[1] > 1
    检查第二个维度是否大于1（表示多分类问题）
    y_hat = y_hat.argmax(axis=1)  # 取每行最大值的索引
    """
    if len(y_hat.shape)>1 and y_hat.shape[1]>1:
        y_hat = y_hat.argmax(axis=1)
    cmp= y_hat.type(y.dtype)==y
    return float(cmp.type(y.dtype).sum())

"""
len(y)
返回张量的第一个维度的大小，相当于 y.shape[0]，主要用于类似数组的访问
y.numel()
返回张量中元素的总个数，适用于任意维度的张量，计算方式是所有维度大小的乘积
"""

def train_epoch(net,train_iter,loss,updater): #返回每个批量的平均损失和平均精度
    # 所有自定义网络都必须继承自 nn.Module
    if isinstance(net,torch.nn.Module):
        net.train()
    for x,y in train_iter: # x y分别是数据集和标签
        y_hat = net(x) # y_hat将是（批量数*特征数量=批量数*通道数*高*宽）
        l = loss(y_hat,y)
        if isinstance(updater,torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(x.shape[0])
    return float(l.sum()/y.numel()), accuracy(y_hat,y)/y.numel()

def plot_training_results(history):
    """
       在PyCharm中绘制训练结果
       """
    # 设置中文字体（可选）
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 绘制损失曲线
    ax1.plot(history['epochs'], history['train_loss'], 'b-o', linewidth=2, markersize=6, label='训练损失')
    ax1.set_xlabel('训练轮次 (Epoch)')
    ax1.set_ylabel('损失值 (Loss)')
    ax1.set_title('训练损失曲线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 绘制准确率曲线
    ax2.plot(history['epochs'], history['train_acc'], 'g-o', linewidth=2, markersize=6, label='训练准确率')
    ax2.plot(history['epochs'], history['test_acc'], 'r-o', linewidth=2, markersize=6, label='测试准确率')
    ax2.set_xlabel('训练轮次 (Epoch)')
    ax2.set_ylabel('准确率 (Accuracy)')
    ax2.set_title('训练和测试准确率曲线')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 调整布局并显示
    plt.tight_layout()

    # 保存图片（可选）
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')

    # 在PyCharm中显示图表
    plt.show()

def train_total(net, train_iter, test_iter, loss, num_epochs, updater):
    # 记录所有epoch的指标
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': [],
        'epochs': list(range(1, num_epochs + 1))
    }
    for epoch in range(num_epochs):
        train_metrics = train_epoch(net,train_iter,loss,updater) #返回每个批量的平均损失、平均训练精度
        test_acc = evaluate_accuracy(net,test_iter)
        train_loss, train_acc = train_metrics
        # 记录历史数据
        # history['epochs'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        # 打印进度
        print(f'Epoch {epoch + 1}/{num_epochs}: '
              f'train_loss={train_loss:.4f}, '
              f'train_acc={train_acc:.4f}, '
              f'test_acc={test_acc:.4f}')
    # 训练结束后绘制图表
    plot_training_results(history)

    return history

# 自己写预测函数
"""
预测步骤：
1.设置评估模式
2.计算正确预测个数、预测总数
3.返回平均预测个数
"""
def evaluate_accuracy(net,test_iter): #返回测试集的平均正确预测个数
    """
    评估模式 (net.eval())：
    Dropout层：所有神经元都参与计算，不进行丢弃
    BatchNorm层：使用训练阶段学到的运行均值/方差进行归一化
    关闭梯度计算：提高推理速度，减少内存占用
    """
    """
    所有自定义网络都必须继承自 nn.Module
    使用 isinstance(net, torch.nn.Module) 可以检查：
    自定义模型类
    各种层（线性层、卷积层等）
    激活函数
    容器（Sequential、ModuleList等）
    任何继承自 nn.Module 的对象
    """
    if isinstance(net,torch.nn.Module):
        net.eval()
    metric = [0.0]*2
    with torch.no_grad():
        for x,y in test_iter:
            metric = [a+float(b) for a,b in zip(metric,(accuracy(net(x),y), y.numel()))]
    return metric[0]/metric[1]

def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat','sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows,num_cols,titles=None,scale = 1.5):
    figsize = (num_rows*scale,num_cols*scale)
    """
    _ 是一个约定俗成的变量名，表示"这个值我不关心"或"忽略这个返回值"
    在这里，它接收 plt.subplots() 返回的 Figure 对象
    因为我们通常更关心子图（axes）而不是图形本身
    """
    _,axes = plt.subplots(num_rows,num_cols,figsize=figsize)
    axes = axes.flatten()
    for i, (ax,img) in enumerate(zip(axes,imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    plt.tight_layout()  # 添加自动调整布局
    plt.show()  # 关键：显示图片
    return axes

def predict_ch4(net,test_iter, n=6):
    for x,y in test_iter:
        break
    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(net(x).argmax(axis=1))
    titles = [true+'\n'+pred for true,pred in zip(trues,preds)]
    show_images(x[0:n].reshape((n,28,28)),1,n,titles=titles[0:n])

train_total(net,train_iter,test_iter,loss,num_epochs,updater)
evaluate_accuracy(net,test_iter)
predict_ch4(net,test_iter)

