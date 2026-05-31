# Experiment: Handwritten RNN Cell
## Experiment Objectives:
1. Understand the recursive structure of RNN: How the hidden state ℎ𝑡 is derived from ℎ𝑡−1 and the input 𝑥𝑡.
2. Handwrite an RNN Cell (without using nn.RNN/nn.RNNCell), mastering the alignment of parameter shapes and dimensions.
3. Train a minimal character-level language model (CharRNN) using the handwritten Cell, observing the decrease in loss and being able to generate text.
## Experiment Data:
tiny_corpus_rnn.txt
## Experiment Content:
1. Data Reading and Character Encoding. Read the text TEXT, construct the character table vocab = sorted(set(TEXT)), and establish stoi/itos mappings. Encode the entire text into an integer sequence ids (length L). Construct next-char training samples (fixed T=32): input x = ids[i : i+T], target y = ids[i+1 : i+T+1]. Print len(TEXT), vocab_size, and a preview of a small portion of TEXT[:200].
2. Embedding Layer. Use Embedding(V, E), recommend E=32. Input (B, T), output (B, T, E). After random batch input, print the shape.
3. Handwritten MyRNNCell. Parameters: Wxh: (E, H); Whh: (H, H); bh: (H,). Forward: Input x_t: (B, E); Input h_prev: (B, H); Output h_t: (B, H). Formula: ℎ𝑡=tanh⁡(??𝑡𝑊𝑥ℎ+ℎ𝑡−1𝑊ℎℎ+??ℎ). Write shape assertion: Ensure the input/output dimensions are completely matched.
4. Time Unrolling (Unroll) to obtain the hidden state sequence. Expand x_emb ((B, T, E)) step by step: h = zeros(B, H) as the initial hidden state; for t in range(T): h = cell(x_emb[:, t, :], h); Save each step h, and finally obtain H_seq: (B, T, H). Need to assert H_seq.shape == (B, T, H).
5. Output Layer and Loss. Output layer: Linear(H, V) (no softmax); logits = linear(H_seq) → (B, T, V); Target: y → (B, T); Loss: Calculate cross-entropy along the time dimension, reshape logits from (B,T,V) to (B*T, V), reshape y from (B,T) to (B*T,), and then use CrossEntropyLoss to calculate.
6. Training Loop, optimizer = Adam(model.parameters(), lr=1e-3), need to do gradient clipping.
7. Text Generation Sampling, implement sample(seed_text, gen_len=200, temperature=1.0). model.eval() + torch.no_grad(), use seed_text to advance the hidden state character by character, get the next character distribution from the last step logits. Selection strategy is random sampling: multinomial(softmax(logits/temperature)), generate 200-character text.
***
# 实验 手写RNN Cell
## 实验目标：
1. 理解 RNN 的递推结构：隐藏状态 ℎ𝑡如何从 ℎ𝑡−1与输入 𝑥𝑡得到。
2. 手写一个 RNN Cell（不使用 nn.RNN/nn.RNNCell），掌握参数形状与维度对齐。
3. 用手写 Cell 训练一个最小字符级语言模型（CharRNN），观察 loss 下降，并能生成文本。
## 实验数据：
tiny_corpus_rnn.txt
## 实验内容：
1. 数据读取与字符编码。
   - 读取文本TEXT ，构建字符表vocab = sorted(set(TEXT))，并建立 stoi/itos 映射。
   - 将全文编码为整数序列 ids（长度 L）。
   - 构造 next-char 训练样本（固定 T=32）：输入 x = ids[i : i+T]，目标 y = ids[i+1 : i+T+1]。
   - 打印 len(TEXT)、vocab_size、以及一小段 TEXT[:200] 预览。 
3. Embedding 层，直接使用Embedding(V, E)，推荐 E=32。输入 (B, T) ，输出 (B, T, E)。随机 batch 输入后，打印shape 
4. 手写 MyRNNCell。参数：Wxh：(E, H)；Whh：(H, H)；bh：(H,)。前向：输入 x_t：(B, E)；输入 h_prev：(B, H)；输出 h_t：(B, H)。公式：ℎ𝑡=tanh⁡(𝑥𝑡𝑊𝑥ℎ+ℎ𝑡−1𝑊ℎℎ+𝑏ℎ) 。写 shape 断言：确保输入/输出维度完全匹配。 
5. 时间展开（Unroll）得到隐藏状态序列。对 x_emb（(B, T, E)）逐步展开：h = zeros(B, H) 作为初始隐藏状态；for t in range(T)：h = cell(x_emb[:, t, :], h)；保存每一步 h，最终得到 H_seq：(B, T, H)。需要assert H_seq.shape == (B, T, H)。 
6. 输出层与损失。输出层：Linear(H, V)（不做 softmax）；logits = linear(H_seq) → (B, T, V)；目标：y → (B, T)；loss：对时间维展开计算交叉熵，将 logits 从 (B,T,V) reshape 为 (B*T, V)，将 y 从 (B,T) reshape 为 (B*T,)，再用 CrossEntropyLoss 计算。 
7. 训练循环，optimizer = Adam(model.parameters(), lr=1e-3)，需要做梯度剪裁。 
8. 文本生成采样 ，实现 sample(seed_text, gen_len=200, temperature=1.0)。model.eval() + torch.no_grad()，用 seed_text 逐字符推进隐藏状态，从最后一步 logits 得到下一字符分布。选择策略为随机采样：multinomial(softmax(logits/temperature))，生成 200 字符文本。

