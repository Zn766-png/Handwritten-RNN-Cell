# 课程作业：12自然语言处理(一)-手写RNN Cell
# 作者：Znn766-png
# 备注：数据来源为项目文件夹内的外部文件tiny_corpus_rnn.txt;
# 实验为手写一个RNN Cell(不使用nn.RNN/nn.RNNCell),用手写Cell训练一个最小字符级语言模型CharRNN并能观察到loss下降和生成文本;
# 训练轮数num_epochs为2000轮

# ===========实验内容0.数据预处理===========
# (1) 导入必要的库
import torch
import torch.nn as nn
from torch.optim import Adam

# ===========实验内容1.数据读取与字符编码===========
# (1) 读取文本TEXT
with open("tiny_corpus_rnn.txt", "r", encoding="utf-8") as f:
    TEXT = f.read()

# (2) 构建字符表vocab = sorted(set(TEXT))
vocab = sorted(set(TEXT))
vocab_size = len(vocab)

# (3) 建立stoi/itos映射
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for ch, i in stoi.items()}

# (4) 将全文编码为整数序列ids(长度 L)
ids = [stoi[ch] for ch in TEXT]     # 需要把文本变成数字列表,因为计算机只认数字

# (5) 构造next-char训练样本(固定T=32):输入x = ids[i : i+T],目标y = ids[i+1 : i+T+1]
T = 32      # next-char是用来根据前T个字符预测下一个字符的
data_x, data_y = [], []     # 初始化两个空列表list,每一对(x,y)就是一个训练样例,x是上下文输入,y   是期望输出。这是监督学习但不是预训练,因为没有下游微调
for i in range(0, len(ids) - T - 1, T):   # 步长T(其实也可以设置为1),且确保有足够长度取y
    data_x.append(ids[i : i+T])     # 切片从i开始的T个数放入data_x列表
    data_y.append(ids[i+1 : i+T+1])     # 切片从i+1开始的T个数放入data_y列表
data_x = torch.tensor(data_x, dtype=torch.long)     # 把列表转换成PyTorch张量,数据类型为整数,形状(N,T)N行T列
data_y = torch.tensor(data_y, dtype=torch.long)     # (N,T)描述张量tensor的维度大小,N样本数量,T每个样本的长度,一批次batch的数据形状都是(N,T)

# (6) 打印len(TEXT)、vocab_size、以及一小段TEXT[:200]预览
print()
print(f"len(TEXT){len(TEXT)}")
print(f"vocab_size{vocab_size}")
print(f"TEXT[:200]预览(部分):\n{TEXT[:200]}")
print()

# ===========实验内容2.Embedding层:直接使用Embedding(V, E),推荐E=32,输入(B, T),输出 (B, T, E),随机batch输入后打印shape===========
# (1) 定义模型超参数
V = vocab_size
E = 32      # 嵌入维度E(嵌入向量长度)
H = 128     # 隐藏状态维度H
B = 64      # batch size

# (2) Embedding层直接使用Embedding(V, E)
embedding = nn.Embedding(V, E)

# (3) 随机batch输入(B,T)后输出(B,T,E)打印shape
dummy_batch = torch.randint(0, V, (B, T))
dummy_emb = embedding(dummy_batch)
print(f"shape:{dummy_emb.shape}")  # 输出是(B, T, E)
print()

# ===========实验内容3.手写MyRNNCell===========
# (1) 创建一个类,定义几个参数并在前向传播方法里实现公式,最后加上assert检查形状
class MyRNNCell(nn.Module):
    def __init__(self, E, H):
        super().__init__()  # 调用父类初始化
        # Cell需要学习的参数初始化(类似神经元间的"突触强度")
        self.W_xh = nn.Parameter(torch.randn(E, H) * 0.01)      # 把输入x_t转换成隐藏状态空间的矩阵,形状是(E, H)
        self.W_hh = nn.Parameter(torch.randn(H, H) * 0.01)      # 把上一时刻的隐藏状态h_prev转换一下的矩阵,形状是(H, H)
        self.b_h = nn.Parameter(torch.zeros(H))     # 偏置项,直接加到结果上形状就是(H,),为了方便广播到(B, H)
    def forward(self, x_t, h_prev):
        assert x_t.dim() == 2 and h_prev.dim() == 2
        assert x_t.shape[1] == self.W_xh.shape[0]   # E
        assert h_prev.shape[1] == self.W_hh.shape[0] # H
        h_t = torch.tanh(x_t @ self.W_xh + h_prev @ self.W_hh + self.b_h)
        assert h_t.shape == (x_t.shape[0], self.W_hh.shape[1])  # (B, H)
        return h_t
class CharRNN(nn.Module):       # 将Cell包装成一个可训练的nn.Module方便优化器和参数收集
    def __init__(self, V, E, H):
        super().__init__()
        self.embedding = nn.Embedding(V, E)
        self.cell = MyRNNCell(E, H)
        self.linear = nn.Linear(H, V)   # 实验内容5(1)输出层Linear(H, V)(不做softmax)
    def forward(self, x):
        B, T = x.shape
        x_emb = self.embedding(x)      # (B, T, E)
        H_seq = []

# ===========实验内容4.时间展开(Unroll)得到隐藏状态序列===========
# (1) 对x_emb((B,T,E))逐步展开:h=zeros(B,H)作为初始隐藏状态;for t in range(T):h=cell(x_emb[:,t,:],h);保存每一步h最终得到H_seq:(B,T,H)
        h = torch.zeros(B, self.cell.W_hh.shape[1], device=x.device)  # h=zeros(B, H)初始化隐藏状态,创建一个全零张量,device确保该张量与输入x在同一个设备(CPU或GPU)上
        for t in range(T):      # for t in range(T):h=cell(x_emb[:,t,:],h)
            h = self.cell.forward(x_emb[:, t, :], h)    # x_emb[:,t,:]取出当前时间步的输入,调用forward传入当前输入和上一个隐藏状态h,计算得到新的隐藏状态h
            H_seq.append(h)     # 将新隐藏状态h添加到列表H_seq中
        H_seq = torch.stack(H_seq, dim=1)    # 将列表H_seq中的张量按照时间维度1堆叠隐藏状态序列,得到形状为(B,T,H)的张量,表示每个时间步的隐藏状态序列,保存每一步h最终得到H_seq:(B,T,H)

# (2) 需要assert H_seq.shape == (B, T, H)
        assert H_seq.shape == (B, T, self.cell.W_hh.shape[1])   # 确保H_seq的形状符合预期(B, T, H)
        logits = self.linear(H_seq)          # 实验内容5(2)logits=linear(H_seq)→(B, T, V),线性变换得到logits将H_seq输入线性层self.linear进行全连接变换,输出形状为(B,T,V)的logits
        return logits, H_seq    # 返回logits和隐藏状态序列H_seq,logits可用于计算交叉熵等损失,H_seq可用于可视化或进一步处理

# ===========实验内容5.输出层与损失===========
# (1) 输出层Linear(H, V)(不做 softmax):在CharRNN里写了
# (2) logits=linear(H_seq)→(B, T, V):在CharRNN里写了
# (3) 目标y→(B, T);loss:对时间维展开计算交叉熵,将logits从(B,T,V)reshape为(B*T, V),将y从(B,T)reshape为(B*T,):在for epoch里写了
#     注:此步把logits和y都平铺成(B*T,)是因为LLM是在每个时间步都做一个分类任务(预测下一个token),所以要把所有位置当成独立样本计算损失
# (4) 用CrossEntropyLoss计算:在criterion里写了
model = CharRNN(V, E, H)    # 实例化模型
print(model)
print()

# ===========实验内容6.训练循环:optimizer=Adam(model.parameters(), lr=1e-3),需要做梯度剪裁===========
# (1) optimizer = Adam(model.parameters(), lr=1e-3)
optimizer = Adam(model.parameters(), lr=1e-3)   # Adam优化器,一种自适应学习率的优化算法,根据梯度的一阶矩和二阶矩动态调整每个参数的更新步长,model把模型中所有可训练参数交给优化器,初始学习率lr=1e-3

# (2) 梯度剪裁(为了在深层Transformer或长序列训练时防止梯度爆炸)与训练循环
criterion = nn.CrossEntropyLoss()       # 实验5(4)用CrossEntropyLoss计算交叉熵损失函数,同时完成softmax和负对数似然计算,输入是未归一化的logits,形状(N,C),目标为类别索引(N,)
def get_batch(data_x, data_y, batch_size):   # 简单随机抽样数据加载器,从总共N条样本中随机抽batch_size个索引,取出对应的输入x和标签y
    N = data_x.shape[0]     # 获取样本总数
    idx = torch.randint(0, N, (batch_size,))    # 在[0, N)范围内随机生成batch_size个整数索引,返回一个形状为(batch_size,)的一维张量
    x = data_x[idx]   # 用PyTorch的整数数组索引(花式索引)从data_x输入序列中抽出idx里每个数字对应的一整行，最终堆叠成一个新的二维张量,形状(batch_size,T),二维张量data_x形状(N,T)
    y = data_y[idx]   # 用PyTorch的整数数组索引(花式索引)从data_y目标序列中抽出idx里每个数字对应的一整行，最终堆叠成一个的二维张量(这是有放回的简单随机抽样)
    return x, y
num_epochs = 2000   # 设置训练轮数2000轮
for epoch in range(1, num_epochs+1):    # 训练循环
    model.train()   # 设置模型为训练模式,启用dropout和batch normalization等训练专用层
    x_batch, y_batch = get_batch(data_x, data_y, B)     # 获取一个批次的数据,调用get_batch函数随机抽取B个样本
    optimizer.zero_grad()   # 清空梯度,将之前计算的梯度清零防止梯度累积
    logits, _ = model(x_batch)    # 前向传播,将输入数据传入模型得到预测结果,logits形状(B,T,V)
    loss = criterion(logits.view(-1, V), y_batch.view(-1))      # 实验内容5(3)目标y的原始形状是(B,T),对时间维展开计算交叉熵将logits从(B,T,V)reshape为(B*T,V),将y从(B,T)reshape为(B*T,),因为PyTorch的nn.CrossEntropyLoss要求input(预测)形状必须是(N,C),target(真实标签)形状必须是(N,)
    loss.backward()     # 反向传播,计算损失关于模型参数的梯度,通过链式法则从输出层反向传播到输入层
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)    # 实验内容6(2)梯度剪裁,计算所有参数的梯度向量的L2范数,如果范数超过max_norm(1.0)则将所有梯度按比例缩放,使缩放后的梯度范数等于max_norm
    optimizer.step()    # 更新参数,根据计算的梯度使用Adam优化算法调整参数值
    if epoch % 200 == 0 or epoch == 1:    # 打印训练信息,每200轮或第1轮打印一次损失值
        print(f"Epoch {epoch:5d} | loss = {loss.item():.4f}")    # 从张量中提取标量损失值
print()

# ===========实验内容7.文本生成采样===========
# (1) 实现sample(seed_text, gen_len=200, temperature=1.0)
@torch.no_grad()    # 进入无梯度计算模式,装饰器表示下面代码无需计算梯度,模型评估或推理阶段节省内存和计算资源
def sample(model, seed_text, gen_len=200, temperature=1.0):     # 实现sample文本生成采样函数,参数model为训练好的CharRNN模型,seed_text为种子文本用于启动生成过程,gen_len为生成文本的长度默认200个字符,temperature本质是放大或压缩词的概率分布,越接近1越倾向选低概率候选词

# (2) model.eval() + torch.no_grad()
    model.eval()    # 设置模型为评估模式,禁用dropout等训练专用层

# (3) 用seed_text逐字符推进隐藏状态
    chars = [stoi.get(ch, 0) for ch in seed_text]   # 编码种子文本,将seed_text转换为索引列表
    x = torch.tensor([chars], dtype=torch.long)   # 将索引列表转换为张量,形状为(1, len_seed)
    x_emb = model.embedding(x)     # 通过嵌入层获取嵌入表示,形状为(1, len_seed, E)
    h = torch.zeros(1, H, device=x.device)   # 初始化隐藏状态形状为(1, H)
    for t in range(x_emb.shape[1]):    # 用seed_text的每个字符推进隐藏状态,遍历seed_text的每个时间步
        h = model.cell.forward(x_emb[:, t, :], h)   # 更新隐藏状态
    last_char_idx = chars[-1]   # 获取seed_text的最后一个字符索引
    generated = list(seed_text)     # 初始化生成的文本列表，包含seed_text
    for _ in range(gen_len):    # 生成指定长度的文本,循环生成gen_len个字符
        x_t = model.embedding(torch.tensor([[last_char_idx]]))  # 将上一步生成的字符索引转换为嵌入,形状为(1, 1, E)
        h = model.cell.forward(x_t[:, 0, :], h)                # 推进隐藏状态,形状(1, H)

# (4) 从最后一步logits得到下一字符分布
        logits = model.linear(h)                               # 从最后一步logits得到下一字符分布,形状(1,V)

# (5) 选择策略为随机采样:multinomial(softmax(logits/temperature)),生成200字符文本
        probs = torch.softmax(logits / temperature, dim=-1).squeeze()   # 温度调节temperature控制分布的平滑程度
        next_idx = torch.multinomial(probs, num_samples=1).item()   # 从概率分布中采样下一个字符索引
        generated.append(itos[next_idx])    # 将生成的字符添加到结果中
        last_char_idx = next_idx    # 更新last_char_idx用于下一次迭代
    return ''.join(generated)   # 返回生成的文本字符串
seed = TEXT[:10] if len(TEXT) >= 10 else "a"    # 选择种子文本,TEXT[:10]取原始文本的前10个字符作为生成文本的"开头",如果总长度不足10就用字母a作为兜底种子
print("seed_text是:",seed)
print("最终结果是:",sample(model, seed, gen_len=200, temperature=0.9))    # 调用生成函数sample并打印,参数是model训练好的CharRNN模型,seed刚才得到的种子开头,gen_len要额外生成的字符个数(最终文本=seed+200个新字符),temperature,print把生成函数返回的完整字符串打印