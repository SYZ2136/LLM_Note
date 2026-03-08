$$
Attention(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

# Encoder

`nn.Linear()`
- **Input**: $(*, H_{in})$ , where $*$ means any number of dimensions including none.
- **Output**: $(*, H_{out})$ , where all but the last dimension are the same shape as the input 
```
self.Wq = nn.Linear(d_model, d_model)
Q = self.Wq(x)  
数学逻辑上，nn.Linear 是矩阵右乘
```

**张量过单个 Encoder layer 层的逻辑**
输入张量 `(B,seq_len,d_model)`，
数学逻辑:  `n_head`组不一样的`Wq,Wk,Wv`右乘,得到`n_head`组 `Q,K,V` , 然后按$\mathrm{scores} = \frac{QK^\top}{\sqrt{d_k}}$ , 计算得`n_head` 个 `scores` 张量，完整自注意力公式计算完后concat 后形状仍为`(B,seq_len,d_model)`
代码实现: 用大的一组`Wq,Wk, Wv`, 形状都为`(d_model, d_model)`
``` 
Q = self.Wq(x)  
K = self.Wk(x)
V = self.Wv(x)  # 形状都为(d_model, d_model)
Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
attn_weights = F.softmax(scores, dim=-1)
```

**Decoder 部分掩码自注意力**
掩码矩阵为下三角矩阵
```
attn_scores = [
    [3.0, 1.0, 2.0, 0.5],
    [0.5, 4.0, 1.5, 2.0],
    [1.0, 0.5, 3.0, 1.5],
    [2.0, 1.5, 0.5, 2.5],
]

mask = [
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [1, 1, 1, 0],
    [1, 1, 1, 1],
]

masked_attn_scores = [
    [3.0,  -1e9,  -1e9,  -1e9],
    [0.5,   4.0,  -1e9,  -1e9],
    [1.0,   0.5,   3.0,  -1e9],
    [2.0,   1.5,   0.5,   2.5],
]  # (B,seq_len,d_model) 过掩码自注意力层形状不变
```

# Decoder-only

**GPT 类的模型，decoder 部分的单个 block 中只有一次掩码自注意力**

进模型前数据为 `(B, seq_len)` 的 token id, embedding 之后得到`(B,seq_len,d_model)`

**训练阶段** 
输入句子前加\<BOS> token，输入 $[\text{BOS}, x_1, \ldots, x_{\text{seq\_len}-1}]$， 标签 $[x_1, \ldots, x_{\text{seq\_len}}]$
$[\text{BOS}, x_1, \ldots, x_{\text{seq\_len}-1}]$ embedding 得 `(B,seq_len,d_model)`，过掩码自注意力块之后还是`(B,seq_len,d_model)` , 算子`nn.(d,vocab_size)` 右乘得`(B,seq_len,vocab_size)`, `vocab_size` 维度上softmax，每一个长 `vocab_size` 的概率向量按照真实标签计算交叉熵

**推理阶段**
\<BOS> token输入，预测出一个 token $x_1$
$[\text{BOS}, x_1]$ 输入，预测 $x_2$ ...
直到预测到结束token
推理阶段有 `KV-cache` 


# Transformer Encoder-Decoder 架构

Decoder 部分的每个 block 除了掩码自注意力机制外，多一次 cross-attention




# 文本和张量的转换过程

### tokenizer 训练

tokenizer 训练是一个独立于 LLM 模型参数训练的前置步骤
```
tokens = list(map(int, text.encode("utf-8")))
# tokens 打印出来就是一个 token id 组成的 list
[239,
 188,
 181,
 239,
 189,
 142,
 239,
 189] 
```

tokenizer 训练的核心产物: 
- 切分规则: `merge`
- 正向映射`vocab(token→id)`
- 反向映射`token→id` (可由vocab反推)
上面三个核心产物可以有不同的数据结构实现，以下给出一种例子：
```
merges
{(240, 159): 256,
 (101, 32): 257,
 (226, 128): 258,
 (97, 110): 259,
 (256, 133): 260,
 (256, 135): 261,
 (239, 189): 262,
 (258, 140): 263,
 (263, 261): 264,
 (115, 32): 265,
 (97, 114): 266,
 (105, 110): 267,
 (32, 259): 268,
 (32, 116): 269,
 (104, 257): 270,
 (100, 32): 271,
 (111, 114): 272,
 (116, 32): 273,
 (101, 114): 274,
 (115, 116): 275}
 
vocab
[('<unk>', 0),
 ('<s>', 1),
 ('</s>', 2),
 ('<0x00>', 3),
 ...
 ('<0xFA>', 253),
 ('<0xFB>', 254),
 ('<0xFC>', 255),
 ('<0xFD>', 256),
 ('<0xFE>', 257),
 ('<0xFF>', 258),
 ('ou', 259),
 ('▁y', 260),
 ('▁you', 261),
 ('▁t', 262),
 ('▁c', 263),
 ('on', 264),
 ('an', 279),]
```

---

### Embedding

本质上就是一个 二维的`nn.Parameter`，形状为`(vocab_size, d_model)`

`tokenizer` 拿到输入文本的 `token_id`后，到 `nn.Parameter` 找出对应行即可，这个过程就是一个简单的张量索引

---

### LLM 尾部文本分类损失计算

交叉熵模块代码使用
```
loss = nn.CrossEntropyLoss()
output = loss(input, target)
# input 形状(N,C) 是 logits，nn.CrossEntropyLoss中封装了softmax的逻辑
# output 形状(N,)
# 表示 N 个 类别个数为 C 的分类任务
```

LLM 尾部相关代码
```
input 取 (B*L, vocab_size)  # 没有softmax，是原始的logits
# input 的每一行就是一个词表大小的分类任务

target 取 (B*L,)
# target 中存的是 B*L 个 token 的相应的 token_id
# 有了token_id, 自然知道标签在词表中的位置，多分类交叉熵对应标签取1，其他地方都是0; 再结合 input 每一行的向量，可以计算得这一个 token 分类任务的交叉熵损失
```

### 推理的随机性


- 用户输入文本 → tokenizer 把文本编码成 token ids。
    
- 这串 token ids 作为 context，模型计算下一个 token 的概率分布 
    
- 直到遇到 EOS/长度上限/停止词等。


**推理时，为什么同一输入输出会不一样？**
**the exploration–exploitation trade-off**
Greedy：每一步取概率最大 argmax。这样同样输入得到同样输出。
实际非贪婪，以及有一些 sampling 的策略

