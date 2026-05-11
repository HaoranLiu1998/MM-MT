#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math, random, pickle, copy, math, os, re, bisect, time, gc, ast, logging
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torch.autograd import Variable
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch.cuda.amp import GradScaler, autocast
from torch_scatter import scatter_mean,scatter_add
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from kan import KANLayer
import numpy as np
from collections import defaultdict
import pandas as pd


# In[2]:


#Set time
def time_elapsed(start_time):
    elapsed = time.time() - start_time  
    hours = int(elapsed/3600)           
    minutes = int(int(elapsed/60)%60)   
    seconds = int(elapsed%60)           
    
    return hours, minutes, seconds


# In[3]:


def get_model_size(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())  # 所有参数占的字节数
    print(f"number of parameters: {param_size}")
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())  # 所有缓冲区占的字节数
    total_size = (param_size + buffer_size) / (1024 ** 2)  # 转换为 MB
    return total_size


# In[4]:


# #Set device
# USE_CUDA = torch.cuda.is_available()
# if USE_CUDA:
#     device = torch.device("cuda")
#     cuda = True
# else:
#     device = torch.device("cpu")
#     cuda = False
    
# print("Device =", device)


# In[5]:


# Set device to GPU 1 if available, otherwise CPU
if torch.cuda.is_available():
    # Always use GPU 1 if it exists
    if torch.cuda.device_count() > 1:
        device = torch.device("cuda:1")
    else:
        device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Device =", device)

# Optional: Set cuda flag for compatibility
cuda = torch.cuda.is_available()


# In[6]:


with open('../data/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

# 提取 stoi 和 itos
stoi = vocab['stoi']
itos = vocab['itos']

# 打印词汇表大小和示例
print("Vocabulary size:", len(stoi))
print("stoi example:", stoi)  # 打印前10个 token
print("itos example:", itos)  # 打印前10个索引对应的 token


# In[7]:


class SMILESDataset(Dataset):
    def __init__(self, smiles_file, props_file, stoi, block_size):
        # 读取 SMILES
        with open(smiles_file, 'r') as f:
            self.smiles = f.read().splitlines()
        
        # 读取属性，跳过第一行表头
        with open(props_file, 'r') as f:
            lines = f.read().splitlines()[1:]
            self.props = [list(map(float, line.strip().split())) for line in lines]
        
        assert len(self.smiles) == len(self.props), "SMILES 和属性数量不匹配！"

        self.stoi = stoi
        self.pad_token = stoi['<pad>']
        self.block_size = block_size

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smiles = self.smiles[idx]
        tokens = re.findall(
            r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])",
            smiles
        )
        tokens = ['<start>'] + tokens + ['<end>']
        tokens = tokens[:self.block_size]
        x = [self.stoi.get(token, self.stoi['<pad>']) for token in tokens]

        if len(x) < self.block_size:
            x += [self.pad_token] * (self.block_size - len(x))
        
        x = torch.tensor(x, dtype=torch.long)
        props = torch.tensor(self.props[idx], dtype=torch.float)
        return x, props


# In[8]:


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        # 两个额外条件 (latent + props)
        num = 2
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size + num, config.block_size + num))
                 .view(1, 1, config.block_size + num, config.block_size + num)
        )
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        attn_save = att
        att = self.attn_drop(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y, attn_save


# In[9]:


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        y, attn = self.attn(self.ln1(x))
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x, attn


# In[10]:


# ===== Main molGPT with Classifier-Free Guidance =====
class molGPT(nn.Module):
    def __init__(self, config, stoi):
        super().__init__()
        self.config = config
        self.vocab_size = len(stoi)
        self.stoi = stoi

        # --- Token + Pos Embeddings ---
        self.tok_emb = nn.Embedding(self.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        # --- Latent + Property Encoder ---
        self.latent_proj = nn.Linear(config.latent_dim, config.n_embd)
        self.prop_proj = nn.Sequential(
            nn.Linear(8, 64),
            nn.GELU(),
            nn.Linear(64, config.n_embd),
        )

        # --- Transformer Backbone ---
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, self.vocab_size, bias=False)
        self.block_size = config.block_size
        self.apply(self._init_weights)

        # --- Fixed normalization range (for 8 properties) ---
        self.register_buffer("prop_min", torch.tensor([100.0, 0, 0, 0, -2, 0, 3, 0]))
        self.register_buffer("prop_max", torch.tensor([900.0, 10, 10, 15, 7, 8, 12, 6]))

    # ===== 初始化权重 =====
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # ===== 属性归一化 =====
    def normalize_props(self, props):
        props = torch.clamp(
            (props - self.prop_min.to(props.device)) /
            (self.prop_max.to(props.device) - self.prop_min.to(props.device)),
            0.0, 1.0
        )
        return props

    # ===== 单次前向传播 =====
    def forward_single(self, idx, props, latent):
        while latent.dim() > 3:
            latent = latent.squeeze(1)  # 去掉多余的维度
        if latent.dim() == 2:
            latent = latent.unsqueeze(1)   # [B, C] → [B, 1, C]
        # 如果 latent 是 [B, L, C] 但 L != 1，则池化成 1 个向量
        if latent.dim() == 3 and latent.size(1) != 1:
            latent = latent.mean(dim=1, keepdim=True)
        
        b, t = idx.size()
        token_embeddings = self.tok_emb(idx)
        position_embeddings = self.pos_emb[:, :t, :]
        x = self.drop(token_embeddings + position_embeddings)

        # --- 条件嵌入 ---
        latent_emb = self.latent_proj(latent)  # (B,1,n_embd)
        prop_emb = self.prop_proj(self.normalize_props(props)).unsqueeze(1)
        
        # 拼接 latent + props + tokens
        x = torch.cat([latent_emb, prop_emb, x], dim=1)

        # --- Transformer 主体 ---
        for layer in self.blocks:
            x, _ = layer(x)
        x = self.ln_f(x)
        logits = self.head(x)
        logits = logits[:, 2:, :]  # skip latent+props
        return logits

    # ===== 支持 Classifier-Free Guidance 的前向传播 =====
    def forward(self, idx, props, targets=None, latent=None, guidance_scale=1.0, unconditional_prob=0.1):
        b, t = idx.size()
        device = idx.device
        if latent is None:
            latent = torch.zeros(b, self.config.latent_dim, device=device)

        # 训练时：随机drop条件（Classifier-Free Guidance 训练阶段）
        if self.training and unconditional_prob > 0:
            mask = (torch.rand(b, 1, device=device) < unconditional_prob).float()
            props = props * (1 - mask)
            latent = latent * (1 - mask)

        # 常规前向
        logits_cond = self.forward_single(idx, props, latent)

        # 生成阶段：双前向推理（条件 + 无条件）
        if not self.training and guidance_scale != 1.0:
            logits_uncond = self.forward_single(
                idx,
                torch.zeros_like(props),
                torch.zeros_like(latent)
            )
            logits = logits_uncond + guidance_scale * (logits_cond - logits_uncond)
        else:
            logits = logits_cond

        # 计算损失
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets.reshape(-1))

        return logits, loss


# In[11]:


#mRNAEncoder submodule
def sinusoidal_position_embedding(device, batch_size = 1, nums_head = 8, max_len = 6400, output_dim = 1024):
    # (max_len, 1)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(-1)
    # (output_dim//2)
    ids = torch.arange(0, output_dim // 2, dtype=torch.float)  
    theta = torch.pow(10000, -2 * ids / output_dim)

    # (max_len, output_dim//2)
    embeddings = position * theta  

    # (max_len, output_dim//2, 2)
    embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

    # (bs, head, max_len, output_dim//2, 2)
    embeddings = embeddings.repeat((batch_size, nums_head, *([1] * len(embeddings.shape))))

    # (bs, head, max_len, output_dim)

    embeddings = torch.reshape(embeddings, (batch_size, nums_head, max_len, output_dim))
    embeddings = embeddings.to(device)
    return embeddings

def RoPE(q, k):
    # q,k: (bs, head, max_len, output_dim)
    batch_size = q.shape[0]
    nums_head = q.shape[1]
    max_len = q.shape[2]
    output_dim = q.shape[-1]

    # (bs, head, max_len, output_dim)
    pos_emb = sinusoidal_position_embedding(q.device, batch_size, nums_head, max_len, output_dim)


    # cos_pos,sin_pos: (bs, head, max_len, output_dim)
    cos_pos = pos_emb[...,  1::2].repeat_interleave(2, dim=-1) 
    sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)

    # q,k: (bs, head, max_len, output_dim)
    q2 = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1)
    q2 = q2.reshape(q.shape)
    q = q * cos_pos + q2 * sin_pos

    k2 = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1)
    k2 = k2.reshape(k.shape)
    k = k * cos_pos + k2 * sin_pos

    return q, k

def attention(q, k, v, mask=None, dropout=None, use_RoPE=True):
    # q.shape: (bs, head, seq_len, dk)
    # k.shape: (bs, head, seq_len, dk)
    # v.shape: (bs, head, seq_len, dk)

    if use_RoPE:
        q, k = RoPE(q, k)

    d_k = k.size()[-1]

    att_logits = torch.matmul(q, k.transpose(-2, -1))  # (bs, head, seq_len, seq_len)
    att_logits /= math.sqrt(d_k)

    if mask is not None:
        att_logits = att_logits.masked_fill(mask == 0, -1e9)

    att_scores = F.softmax(att_logits, dim=-1)  # (bs, head, seq_len, seq_len)

    if dropout is not None:
        att_scores = dropout(att_scores)

    # (bs, head, seq_len, seq_len) * (bs, head, seq_len, dk) = (bs, head, seq_len, dk)
    return torch.matmul(att_scores, v), att_scores

# q = torch.randn((8, 12, 10, 32))
# k = torch.randn((8, 12, 10, 32))
# v = torch.randn((8, 12, 10, 32))

# q = q.to(device)
# k = k.to(device)
# v = v.to(device)

# res, att_scores = attention(q, k, v, mask=None, dropout=None, use_RoPE=True)


#     # (bs, head, seq_len, dk),  (bs, head, seq_len, seq_len)
# print(res.shape, att_scores.shape)


# In[12]:


class mRNAEncoder (nn.Module):
    def __init__(self):
        super(mRNAEncoder, self).__init__()
        self.Conv_1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=1, stride=1)
        self.Conv_2 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=1, stride=3)
        self.Conv_3 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=1, stride=3)
        self.Conv_4 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=1, stride=3)
        self.Conv_5 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=3)
        self.Conv_6 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=9, stride=3,padding=3)
        #ninp=1024, nhead=8, nhid=1024, nlayers=3, dropout=0.2
        d_model = 4*3+4*3+64+128
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=1024, dropout=0.2, batch_first=True)
        self.mRNA_transformer_encoder = TransformerEncoder(copy.deepcopy(encoder_layer), num_layers=4)
        self.outlinear = nn.Linear(4*3+4*3+64+128,1024)
        
    def forward(self,mRNA_seq):
        mRNA_seq = mRNA_seq[:((mRNA_seq.size()[0] - 1) // 3 * 3)].reshape(1,-1).float()
        out_1 = self.Conv_1(mRNA_seq)
        out_1_0 = out_1[0,:].reshape(-1,3).t()
        out_1_1 = out_1[1,:].reshape(-1,3).t()
        out_1_2 = out_1[2,:].reshape(-1,3).t()
        out_1_3 = out_1[3,:].reshape(-1,3).t()
        out_2 = self.Conv_2(mRNA_seq)
        out_3 = self.Conv_3(mRNA_seq[:,1:])
        out_4 = self.Conv_4(mRNA_seq[:,2:])        
        out_5 = self.Conv_5(mRNA_seq)
        out_6 = self.Conv_6(mRNA_seq)
        #168264
        cnn_out = torch.cat([out_1_0, out_1_1, out_1_2, out_1_3, out_2, out_3, out_4, out_5, out_6], 0)
        fcnn_out = cnn_out.t()
        RoPE_in = fcnn_out.reshape(1,1,-1,4*3+4*3+64+128)
        RoPE_out, att_scores = attention(RoPE_in, RoPE_in, RoPE_in, mask=None, dropout=None, use_RoPE=True)
        RoPE_out = RoPE_out.squeeze(0)##(batch,seq_len,ninp)
        
        transf_out = self.mRNA_transformer_encoder(RoPE_out)#(batch,seq_len,hidden_zise)
        mRNA_vector = torch.unsqueeze(torch.mean(transf_out, 1), 0)#(batch,1,hidden_zise)
        mRNA_vector = self.outlinear(mRNA_vector).reshape(1,1024)
        
        return mRNA_vector


# In[13]:


# -------------------------
# Lightweight attention conv using KAN for node projection
# -------------------------
class LightGATConv(MessagePassing):
    def __init__(self, in_dim, out_dim, edge_dim=3, aggr='add'):
        super().__init__(aggr=aggr)
        # KANLayer with grid=2 (very lightweight)
        self.node_proj = KANLayer(in_dim, out_dim)
        # edge projection remains cheap linear
        self.edge_proj = nn.Linear(edge_dim, out_dim)
        self.out_dim = out_dim
        self.scale = out_dim ** 0.5

    def forward(self, x, edge_index, edge_attr):
        # Node projection (use only first output)
        x_proj = self.node_proj(x)[0]       # [N, out_dim] KAN
        # Edge projection
        e_proj = self.edge_proj(edge_attr)  # [E, out_dim] Linear

        return self.propagate(edge_index, x=x_proj, edge_attr=e_proj)

    def message(self, x_i, x_j, edge_attr, index):
        #x_i = x[col], x_j = x[row], edge_attr = e_proj
        #row = edge_index[0], index = col = edge_index[1]
        # dot-product attention
        score = (x_i * x_j).sum(dim=-1) / self.scale

        # edge bias (cheap)
        edge_bias = (edge_attr.sum(dim=-1)) / (edge_attr.size(-1) ** 0.5)
        score = F.leaky_relu(score + edge_bias, negative_slope=0.2)

        alpha = softmax(score, index)
        return x_j * alpha.unsqueeze(-1)


# In[14]:


# -------------------------
# Light GNN block
# -------------------------
class LightweightBlock(nn.Module):
    def __init__(self, dim, edge_dim=3, dropout=0.1):#dim=hidden_dim
        super().__init__()
        self.conv = LightGATConv(dim, dim, edge_dim=edge_dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr):
        h = self.conv(x, edge_index, edge_attr)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.norm(h + x)  # residual + LN


# In[15]:


# -------------------------
# Lightweight ProteinGNNEncoder_KAN (grid=2 everywhere)
# -------------------------
class ProteinGNNEncoder_KAN_Light(nn.Module):
    def __init__(self,in_dim=20,hidden_dim=256,latent_dim=1024,edge_dim=3,dropout=0.1):
        super().__init__()
        # lightweight KAN input projection
        self.kan_in = KANLayer(in_dim, hidden_dim)
        self.block1 = LightweightBlock(hidden_dim, edge_dim=edge_dim, dropout=dropout)
        self.block2 = LightweightBlock(hidden_dim, edge_dim=edge_dim, dropout=dropout)
        # pooling
        self.pool_mean = global_mean_pool
        self.pool_max = global_max_pool
        # small KAN after pooling
        pooled_dim = hidden_dim * 2
        reduced_dim = min(pooled_dim, 128)
        self.kan_pool = KANLayer(pooled_dim, reduced_dim)
        self.fc_out = nn.Sequential(
            nn.Linear(reduced_dim, latent_dim),
            nn.ReLU(),
            nn.LayerNorm(latent_dim)
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch if hasattr(data, "batch") else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        # KAN input projection
        x = self.kan_in(x)[0]
        x = self.block1(x, edge_index, edge_attr)
        x = self.block2(x, edge_index, edge_attr)
        # pooling
        x_mean = self.pool_mean(x, batch)
        x_max = self.pool_max(x, batch)
        graph_feat = torch.cat([x_mean, x_max], dim=1)
        # KAN pooling
        pooled = self.kan_pool(graph_feat)[0]
        latent = self.fc_out(pooled)
        return latent


# In[16]:


class Targetfeaturefusion(nn.Module):
    def __init__(self, ninp = 1024, nhead = 8, nhid= 1024, nlayers= 1, dropout=0.2):
        super(Targetfeaturefusion, self).__init__()
        self.kan1 = KANLayer(in_dim=ninp, out_dim=nhid)
        self.kan2 = KANLayer(in_dim=ninp, out_dim=nhid)
        self.kan3 = KANLayer(in_dim=ninp, out_dim=nhid)
        
    def forward(self,gene_fv, protstru_fv, protemb_fv):
        protemb_fv = protemb_fv.reshape(-1,1024)
        # print(f'gene_fv size: {gene_fv.size()} ')
        # print(f'protstru_fv size: {protstru_fv.size()} ')
        # print(f'protemb_fv size: {protemb_fv.size()} ')
        genekanout = self.kan1(gene_fv)[0]
        protskanout = self.kan2(protstru_fv)[0]
        protekanout = self.kan3(protemb_fv)[0]
        TargetfusionFeature, att_scores = attention(genekanout, protskanout, protekanout, mask=None, dropout=None,use_RoPE = False)
        #print("FusionFeature size: ",FusionFeature.size())
        FusionFeature = TargetfusionFeature.reshape(-1,1,1024)
        
        return FusionFeature


# In[17]:


class Multitargetfeaturefusion(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=1024, dropout=0.1):
        super().__init__()
        self.kan1 = KANLayer(in_dim=input_dim*2, out_dim=hidden_dim)
        #self.kan2 = KANLayer(in_dim=input_dim, out_dim=hidden_dim)
        # 特征融合层：将两个向量拼接后映射到隐藏维度
        #self.fc1 = nn.Linear(2 * input_dim, hidden_dim)
        # 非线性激活函数（GELU效果通常优于ReLU）
        #self.activation = nn.GELU()
        # 最终映射回输入维度
        #self.fc2 = nn.Linear(hidden_dim, input_dim)
        # 归一化层和Dropout防止过拟合
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)           
    
    def forward(self, x1, x2):
        # 拼接两个特征向量 [1, 1024] -> [1, 2048]
        x1 = x1.reshape(-1,1024)
        x2 = x2.reshape(-1,1024)
        x = torch.cat([x1, x2], dim=-1)
        # 特征融合和非线性变换
        x = self.kan1(x)[0]          # [1, 2048] -> [1, 2048]
        #x = self.kan2(x)[0]           # [1, 2048] -> [1, 1024]
        x = self.dropout(x)       # 随机失活
        # 残差连接 + 归一化
        x, att_scores = attention(x, x, x, mask=None, dropout=None,use_RoPE = False)
        x = self.norm(x + (x1 + x2)/2)  # 残差连接均值
        x = x.reshape(-1,1,1024)
        return x


# In[18]:


class MMMTDD(nn.Module):
    def __init__(self, molGPTconfig, smiles_stoi):
        super(MMMTDD, self).__init__()

        # === encoders ===
        self.gencoder = mRNAEncoder()
        self.protencoder = ProteinGNNEncoder_KAN_Light()

        # === fusion ===
        self.targetfeaturefusion = Targetfeaturefusion()
        self.Multitargetfeaturefusion = Multitargetfeaturefusion()

        # === decoder ===
        self.drugdecoder = molGPT(molGPTconfig, stoi=smiles_stoi)

    # @staticmethod
    # def debug_shape(name, x):
    #     try:
    #         print(f"{name}: {tuple(x.shape)}")
    #     except:
    #         pass

    # ===================================================
    # forward
    # ===================================================
    def forward(self, DrugG_in, DrugG_outpt, mol_prop,
                T1gene, T1protstru, T1protemb_fv,
                T2gene, T2protstru, T2protemb_fv):

        # ===== encode target 1 =====
        T1gene_fv = self.gencoder(T1gene)
        T1prot_fv = self.protencoder(T1protstru)
        T1multi = self.targetfeaturefusion(T1gene_fv, T1prot_fv, T1protemb_fv)

        # ===== encode target 2 =====
        T2gene_fv = self.gencoder(T2gene)
        T2prot_fv = self.protencoder(T2protstru)
        T2multi = self.targetfeaturefusion(T2gene_fv, T2prot_fv, T2protemb_fv)

        # ===== fuse two targets =====
        Multi_T_Fv = self.Multitargetfeaturefusion(T1multi, T2multi)

        # ===== debug =====
        #self.debug_shape("T1multi", T1multi)
        #self.debug_shape("T2multi", T2multi)
        #self.debug_shape("Multi_T_Fv", Multi_T_Fv)

        # ===== safety =====
        Multi_T_Fv = torch.nan_to_num(Multi_T_Fv)
        Multi_T_Fv = torch.clamp(Multi_T_Fv, -1e3, 1e3)

        # ===== decoder =====
        logits, loss = self.drugdecoder(
            DrugG_in,
            props=mol_prop,
            targets=DrugG_outpt,
            latent=Multi_T_Fv
        )

        return logits, loss

    # ===================================================
    # optimizer
    # ===================================================
    def configure_optimizers(self, train_config):

        decay = set()
        no_decay = set()

        whitelist_weight = (nn.Linear, nn.LSTM)
        blacklist_weight = (nn.LayerNorm, nn.Embedding)

        param_dict = {pn: p for pn, p in self.named_parameters()}

        # ===== 分类参数 =====
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters(recurse=False):
                fpn = f"{mn}.{pn}" if mn else pn

                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight):
                    no_decay.add(fpn)
                else:
                    decay.add(fpn)

        inter = decay & no_decay
        assert len(inter) == 0, f"交叉: {inter}"

        # ===== 基本组 =====
        optim_groups = [
            {
                "params": [param_dict[p] for p in sorted(decay)],
                "weight_decay": train_config.weight_decay
            },
            {
                "params": [param_dict[p] for p in sorted(no_decay)],
                "weight_decay": 0.0
            },
        ]

        # ===== 双学习率模块 =====
        double_lr_modules = [
            self.protencoder,
            self.targetfeaturefusion,
            self.Multitargetfeaturefusion,
        ]

        # 用 id 处理避免 tensor 比较
        double_lr_params = []
        for m in double_lr_modules:
            double_lr_params.extend(list(m.parameters()))
        double_lr_param_ids = {id(p) for p in double_lr_params}

        # 移除旧组内重复参数
        for group in optim_groups:
            group["params"] = [p for p in group["params"] if id(p) not in double_lr_param_ids]

        # 添加新组
        optim_groups.append({
            "params": double_lr_params,
            "lr": train_config.learning_rate * 0.5,
            "weight_decay": train_config.weight_decay,
        })

        # ===== optimizer =====
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=train_config.learning_rate,
            betas=train_config.betas,
        )

        return optimizer


# In[19]:


class GPTConfig:
    def __init__(self, vocab_size, block_size, latent_dim, **kwargs):
        # ===== 基础结构参数 =====
        self.vocab_size = vocab_size         # 词表大小
        self.block_size = block_size         # 序列最大长度
        self.latent_dim = latent_dim         # 潜变量维度

        # ===== 模型维度 =====
        self.n_layer = kwargs.get("n_layer", 8)
        self.n_head = kwargs.get("n_head", 8)
        self.n_embd = kwargs.get("n_embd", 256)

        # ===== dropout 参数 =====
        self.embd_pdrop = kwargs.get("embd_pdrop", 0.1)
        self.resid_pdrop = kwargs.get("resid_pdrop", 0.1)
        self.attn_pdrop = kwargs.get("attn_pdrop", 0.1)

        # ===== 属性参数 =====
        self.num_props = kwargs.get("num_props", 8)  # 属性数量固定为8
        self.use_properties = kwargs.get("use_properties", True)

        # ===== Classifier-Free Guidance 参数 =====
        self.guidance_scale = kwargs.get("guidance_scale", 1.0)  # 推理时条件放大倍数
        self.unconditional_prob = kwargs.get("unconditional_prob", 0.1)  # 训练时条件drop概率

        # ===== 其他扩展项（可选） =====
        self.use_latent = kwargs.get("use_latent", True)
        self.initializer_range = kwargs.get("initializer_range", 0.02)

        # 允许任意其他参数覆盖
        for k, v in kwargs.items():
            setattr(self, k, v)


# In[20]:


# ===== 创建配置实例 =====
config = GPTConfig(
    vocab_size=len(stoi),
    block_size=128,
    latent_dim=1024,
    n_layer=8,
    n_head=8,
    n_embd=256,
    num_props=8,
    guidance_scale=1.0,        # 推理时使用 1.0 表示普通前向；>1 启用 CFG
    unconditional_prob=0.1,    # 训练时 10% 的样本随机去掉条件
)


# In[21]:


MMMTDDmodelv5 = MMMTDD(config,stoi)
print(f"Model size: {get_model_size(MMMTDDmodelv5):.2f} MB")
print("Submodule：")
for name, submodule in MMMTDDmodelv5.named_children():
    num_params = sum(p.numel() for p in submodule.parameters())
    print(f"{name}: {num_params} parameters")


# In[22]:


class MultiModalDualTargetDataset(Dataset):
    def __init__(self, csv_file, smiles_stoi, smiles_block_size, 
                 prot_emb_dir, prot_struct_dir,
                 mol_prop_file="./filtered_dual_mol_8properties.txt"):
        """
        新增:
          mol_prop_file: 每个分子的 8 个属性
        """
        self.data = pd.read_csv(csv_file)
        self.stoi = smiles_stoi
        self.pad_token = smiles_stoi['<pad>']
        self.block_size = smiles_block_size
        self.prot_emb_dir = prot_emb_dir
        self.prot_struct_dir = prot_struct_dir

        # ----------- 新增：加载分子 8 属性 -----------
        # 跳过表头，按行读取
        self.mol_props = pd.read_csv(mol_prop_file, sep="\t")  # 自动用表头
        assert len(self.mol_props) == len(self.data), \
            "ERROR: filtered_dual_mol_8properties.txt 行数必须与 CSV 对齐！"

        # 转为 torch float32
        self.mol_props = torch.tensor(self.mol_props.values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def tokenize_smiles(self, smiles):
        tokens = re.findall(r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])", smiles)
        tokens = ['<start>'] + tokens + ['<end>']
        tokens = tokens[:self.block_size]
        x = [self.stoi.get(token, self.stoi.get('<unk>')) for token in tokens]
        if len(x) < self.block_size:
            x = x + [self.pad_token] * (self.block_size - len(x))
        return torch.tensor(x, dtype=torch.long)

    def tokenize_gene_seq(self, gene_seq_str):
        tokens = ast.literal_eval(gene_seq_str)
        return torch.tensor(tokens, dtype=torch.long)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # ---------------- SMILES ----------------
        smiles_tensor = self.tokenize_smiles(row['SMILES'])

        # ----------- 新增：分子属性（8 维） -----------
        mol_prop_tensor = self.mol_props[idx]  # shape: [8]

        # ---------------- 基因序列 ----------------
        gene_seq1 = self.tokenize_gene_seq(row['Target1_Encoded_Gene_Seq'])
        gene_seq2 = self.tokenize_gene_seq(row['Target2_Encoded_Gene_Seq'])

        # ---------------- 蛋白质数据 ----------------
        target1 = row['Target1_Uniprot']
        target2 = row['Target2_Uniprot']

        emb1_path = os.path.join(self.prot_emb_dir, f"{target1}.npy")
        emb2_path = os.path.join(self.prot_emb_dir, f"{target2}.npy")
        struct1_path = os.path.join(self.prot_struct_dir, f"alphafold_{target1}_surf.pt")
        struct2_path = os.path.join(self.prot_struct_dir, f"alphafold_{target2}_surf.pt")

        prot_emb1 = torch.tensor(np.load(emb1_path), dtype=torch.float32)
        prot_emb2 = torch.tensor(np.load(emb2_path), dtype=torch.float32)
        prot_struct1 = torch.load(struct1_path)
        prot_struct2 = torch.load(struct2_path)

        return {
            'smiles': smiles_tensor,
            'mol_prop': mol_prop_tensor,   # <----------- 新增模态
            'gene_seq1': gene_seq1,
            'prot_struct1': prot_struct1,
            'prot_emb1': prot_emb1,
            'gene_seq2': gene_seq2,
            'prot_struct2': prot_struct2,
            'prot_emb2': prot_emb2
        }


def custom_collate(batch):
    batch_collated = {}

    # SMILES
    batch_collated['smiles'] = torch.stack([item['smiles'] for item in batch], dim=0)

    # ---------- 新增：堆叠分子属性 ----------
    batch_collated['mol_prop'] = torch.stack([item['mol_prop'] for item in batch], dim=0)

    # 核苷酸序列保持 list
    batch_collated['gene_seq1'] = [item['gene_seq1'] for item in batch]
    batch_collated['gene_seq2'] = [item['gene_seq2'] for item in batch]

    # 蛋白质 embedding
    batch_collated['prot_emb1'] = torch.stack([item['prot_emb1'] for item in batch], dim=0)
    batch_collated['prot_emb2'] = torch.stack([item['prot_emb2'] for item in batch], dim=0)

    # 图数据
    batch_collated['prot_struct1'] = Batch.from_data_list([item['prot_struct1'] for item in batch])
    batch_collated['prot_struct2'] = Batch.from_data_list([item['prot_struct2'] for item in batch])

    return batch_collated


# In[23]:


dataset = MultiModalDualTargetDataset(
    csv_file="../data/multidata/filtered_dual_target_data2.csv",
    smiles_stoi=stoi,
    smiles_block_size=128,
    prot_emb_dir="../data/multidata/prot_emb_data",
    prot_struct_dir="../data/multidata/stru_data",
    mol_prop_file="../data/multidata/filtered_dual_mol_8properties.txt"
)


# In[24]:


dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=custom_collate)
for batch in dataloader:
    print("\n====== New batch ======")
    print(batch)  # 整个字典结构
    # -------- SMILES ----------
    print("\nSMILES tokens:")
    print(batch['smiles'])
    # -------- 分子属性 ----------
    print("\nmol_prop:")
    print(batch['mol_prop'].shape)  # [8]
    print(batch['mol_prop'])
    # -------- Target 1 ----------
    print("\nTarget 1:")
    print("gene_seq1 size:", batch['gene_seq1'][0].size())
    print(batch['gene_seq1'])
    print("prot_struct1:")
    print(batch['prot_struct1'])
    print("prot_emb1 size:", batch['prot_emb1'].size())
    print(batch['prot_emb1'])
    # -------- Target 2 ----------
    print("\nTarget 2:")
    print("gene_seq2 size:", batch['gene_seq2'][0].size())
    print(batch['gene_seq2'])
    print("prot_struct2:")
    print(batch['prot_struct2'])
    print("prot_emb2 size:", batch['prot_emb2'].size())
    print(batch['prot_emb2'])
    break


# In[25]:


def train(model, loader, optimizer, config, device='cuda'):
    model.train()
    scaler = GradScaler()
    tokens_processed = 0

    for epoch in range(1, config.max_epochs + 1):
        pbar = tqdm(enumerate(loader), total=len(loader))
        batchno = 0
        batchsave = 0
        for it, batch in pbar:
            # -------------------------
            # 取出 batch 数据
            smiles_tensor = batch['smiles'].to(device)                  # [B, L]
            mol_prop      = batch['mol_prop'].to(device)                # [B, 8]
            target1_nuc   = batch['gene_seq1'][0].to(device)            # variable length
            target2_nuc   = batch['gene_seq2'][0].to(device)
            # ⚠️ 图数据是 PyG Batch，不要取 [0]
            target1_graph = batch['prot_struct1'].to(device)
            target2_graph = batch['prot_struct2'].to(device)
            target1_emb   = batch['prot_emb1'].to(device)
            target2_emb   = batch['prot_emb2'].to(device)
            # -------------------------
            # 语言模型输入（shift one token）
            inpt  = smiles_tensor[:, :-1]
            outpt = smiles_tensor[:, 1:]
            optimizer.zero_grad()
            with autocast():
                # -------------------------
                # 新增 mol_prop 输入
                logits, loss = model(
                    inpt, outpt,
                    mol_prop,
                    target1_nuc, target1_graph, target1_emb,
                    target2_nuc, target2_graph, target2_emb
                )
                loss = loss.mean()
            if torch.isnan(loss) or torch.isinf(loss):
                print("❌ Loss NaN/Inf, skip batch.")
                continue
            # -------------------------
            # backward
            try:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               config.grad_norm_clip)
                scaler.step(optimizer)
                scaler.update()

            except RuntimeError as e:
                print(f"⚠️ Runtime error during backward: {e}")
                continue
            # -------------------------
            # (可选) 学习率衰减
            if config.lr_decay:
                tokens_processed += (outpt >= 0).sum()
                if tokens_processed < config.warmup_tokens:
                    lr_mult = tokens_processed / max(1, config.warmup_tokens)
                else:
                    progress = (tokens_processed - config.warmup_tokens) / \
                               max(1, config.final_tokens - config.warmup_tokens)
                    lr_mult = max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))
                lr = config.learning_rate * lr_mult
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
            else:
                lr = config.learning_rate
            pbar.set_description(
                f"Epoch {epoch}, Iter {it}: Loss {loss.item():.5f}, LR {lr:.6f}"
            )
            # 保存中间 checkpoint
            batchno += 1
            if batchno == len(loader) // 2:
                torch.save(model.state_dict(),
                           f"MMMTDDmodelv5_epoch_{epoch}_batch_{batchsave+1}.pth")
                batchsave += 1
                batchno = 0
        # 保存 epoch 结束的 checkpoint
        torch.save(model.state_dict(), f"MMMTDDmodelv5_epoch_{epoch}.pth")


# In[26]:


class TrainerConfig:
    def __init__(self, **kwargs):
        # 优化参数
        self.max_epochs = 10  # 最大训练 epoch 数
        self.batch_size = 1  # 批量大小
        self.latent_dim = 1024
        self.learning_rate = 2e-6  # 学习率
        self.betas = (0.9, 0.95)  # Adam 优化器的动量参数
        self.weight_decay = 0.1  # 权重衰减
        self.grad_norm_clip = 0.8  # 梯度裁剪阈值
        # 学习率衰减参数
        self.lr_decay = False  # 是否启用学习率衰减
        self.warmup_tokens = 375e6  # warmup 的 token 数
        self.final_tokens = 260e9  # 学习率衰减到 10% 的 token 数

        # 检查点设置
        self.ckpt_path = None  # 模型保存路径
        self.num_workers = 0  # 数据加载器的线程数

        # 其他参数
        for k, v in kwargs.items():
            setattr(self, k, v)


# In[27]:


# 创建训练配置
train_config = TrainerConfig(
    max_epochs=20,
    batch_size=1,
    learning_rate=4e-1,
    weight_decay=0.1,
    grad_norm_clip=1.0,
    ckpt_path='MMMTDDmodelv5'
)

# 初始化优化器
optimizer = MMMTDDmodelv5.configure_optimizers(train_config)


# In[28]:


checkpoint_path = "modelv10.pt_epoch_5.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu")
# 获取 drugdecoder 当前 state_dict
drugdecoder_state_dict = MMMTDDmodelv5.drugdecoder.state_dict()
# 过滤 checkpoint，只保留匹配的 key
filtered_state_dict = {}
for k, v in checkpoint.items():
    if k in drugdecoder_state_dict:
        filtered_state_dict[k] = v
missing_keys, unexpected_keys = MMMTDDmodelv5.drugdecoder.load_state_dict(filtered_state_dict, strict=False)

print("=== DrugDecoder Weight Load Report ===")
print("Missing keys (not loaded, will be initialized):", missing_keys)
print("Unexpected keys (in checkpoint but not in model):", unexpected_keys)
print("=====================================")
for name, param in MMMTDDmodelv5.drugdecoder.named_parameters():
    if param.requires_grad:
        # 如果 param 全是 0 或 NaN/Inf，就初始化
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"Re-initializing unsafe param: {name}")
            nn.init.normal_(param, mean=0.0, std=0.02)

def clamp_drugdecoder_outputs(module, min_val=-1e3, max_val=1e3):
    for name, param in module.named_parameters():
        if param.requires_grad:
            with torch.no_grad():
                param.clamp_(min_val, max_val)
                
clamp_drugdecoder_outputs(MMMTDDmodelv5.drugdecoder)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MMMTDDmodelv5 = MMMTDDmodelv5.to(device)

print("Checkpoint safely loaded and model moved to device.")


# In[ ]:


train(MMMTDDmodelv5,dataloader, optimizer, train_config, device=device)


# In[ ]:




