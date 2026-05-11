#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""
import pickle
import math
import logging
import re
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import math
import gc
from torch.cuda.amp import autocast, GradScaler


# In[2]:


#Set time
def time_elapsed(start_time):
    elapsed = time.time() - start_time  
    hours = int(elapsed/3600)           
    minutes = int(int(elapsed/60)%60)   
    seconds = int(elapsed%60)           
    
    return hours, minutes, seconds


# In[3]:


with open('../data/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

stoi = vocab['stoi']
itos = vocab['itos']

print("Vocabulary size:", len(stoi))
print("stoi example:", stoi)  # 打印前10个 token
print("itos example:", itos)  # 打印前10个索引对应的 token


class SMILESDataset(Dataset):
    def __init__(self, smiles_file, props_file, stoi, block_size):
        with open(smiles_file, 'r') as f:
            self.smiles = f.read().splitlines()

        with open(props_file, 'r') as f:
            lines = f.read().splitlines()[1:]
            self.props = [list(map(float, line.strip().split())) for line in lines]
        
        assert len(self.smiles) == len(self.props), "SMILES and attribute quantity do not match！"

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


# In[6]:


smiles_file = '../data/pretrainedsmiles.txt'
props_file  = '../data/pretrainedsmiles_8properties.txt'
block_size = 128

dataset = SMILESDataset(smiles_file, props_file, stoi, block_size)
loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True, persistent_workers=False)

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

class molGPT(nn.Module):
    def __init__(self, config, stoi):
        super().__init__()
        self.config = config
        self.vocab_size = len(stoi)
        self.stoi = stoi

        self.tok_emb = nn.Embedding(self.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        self.latent_proj = nn.Linear(config.latent_dim, config.n_embd)
        self.prop_proj = nn.Sequential(
            nn.Linear(8, 64),
            nn.GELU(),
            nn.Linear(64, config.n_embd),
        )

        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, self.vocab_size, bias=False)
        self.block_size = config.block_size
        self.apply(self._init_weights)

        self.register_buffer("prop_min", torch.tensor([0, 0, 0, 0, -2, 0, 3, 0]))
        self.register_buffer("prop_max", torch.tensor([500, 5, 5, 8, 4, 4, 8, 3]))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def normalize_props(self, props):
        props = torch.clamp(
            (props - self.prop_min.to(props.device)) /
            (self.prop_max.to(props.device) - self.prop_min.to(props.device)),
            0.0, 1.0
        )
        return props

    def configure_optimizers(self, train_config):
        decay, no_decay = set(), set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.LSTM)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith('bias') or ('bias' in pn):
                    no_decay.add(fpn)
                elif (pn.endswith('weight') or ('weight' in pn)) and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        no_decay.add('pos_emb')

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))],
             "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=train_config.learning_rate,
            betas=train_config.betas
        )
        return optimizer

    def forward_single(self, idx, props, latent):
        b, t = idx.size()
        token_embeddings = self.tok_emb(idx)
        position_embeddings = self.pos_emb[:, :t, :]
        x = self.drop(token_embeddings + position_embeddings)

        latent_emb = self.latent_proj(latent).unsqueeze(1)  # (B,1,n_embd)
        prop_emb = self.prop_proj(self.normalize_props(props)).unsqueeze(1)

        x = torch.cat([latent_emb, prop_emb, x], dim=1)

        for layer in self.blocks:
            x, _ = layer(x)
        x = self.ln_f(x)
        logits = self.head(x)
        logits = logits[:, 2:, :]  # skip latent+props
        return logits

    def forward(self, idx, props, targets=None, latent=None, guidance_scale=1.0, unconditional_prob=0.1):
        b, t = idx.size()
        device = idx.device
        if latent is None:
            latent = torch.zeros(b, self.config.latent_dim, device=device)

        if self.training and unconditional_prob > 0:
            mask = (torch.rand(b, 1, device=device) < unconditional_prob).float()
            props = props * (1 - mask)
            latent = latent * (1 - mask)

        logits_cond = self.forward_single(idx, props, latent)

        if not self.training and guidance_scale != 1.0:
            logits_uncond = self.forward_single(
                idx,
                torch.zeros_like(props),
                torch.zeros_like(latent)
            )
            logits = logits_uncond + guidance_scale * (logits_cond - logits_uncond)
        else:
            logits = logits_cond

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets.reshape(-1))

        return logits, loss


class GPTConfig:
    def __init__(self, vocab_size, block_size, latent_dim, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.latent_dim = latent_dim

        self.n_layer = kwargs.get("n_layer", 8)
        self.n_head = kwargs.get("n_head", 8)
        self.n_embd = kwargs.get("n_embd", 256)

        self.embd_pdrop = kwargs.get("embd_pdrop", 0.1)
        self.resid_pdrop = kwargs.get("resid_pdrop", 0.1)
        self.attn_pdrop = kwargs.get("attn_pdrop", 0.1)

        self.num_props = kwargs.get("num_props", 8)
        self.use_properties = kwargs.get("use_properties", True)

        self.guidance_scale = kwargs.get("guidance_scale", 1.0)  # 推理时条件放大倍数
        self.unconditional_prob = kwargs.get("unconditional_prob", 0.1)  # 训练时条件drop概率

        self.use_latent = kwargs.get("use_latent", True)
        self.initializer_range = kwargs.get("initializer_range", 0.02)

        for k, v in kwargs.items():
            setattr(self, k, v)

config = GPTConfig(
    vocab_size=len(stoi),
    block_size=128,
    latent_dim=1024,
    n_layer=8,
    n_head=8,
    n_embd=256,
    num_props=8,
    guidance_scale=1.0,
    unconditional_prob=0.1,
)

modelv10 = molGPT(config, stoi=stoi).to("cuda")
print("number of modelv10 parameters: %e", sum(p.numel() for p in modelv10.parameters()))


class TrainerConfig:
    def __init__(self, **kwargs):
        self.max_epochs = 10
        self.batch_size = 128
        self.latent_dim = 1024
        self.learning_rate = 1e-5
        self.betas = (0.9, 0.95)
        self.weight_decay = 0.1
        self.grad_norm_clip = 1.0
        self.lr_decay = False
        self.warmup_tokens = 375e6
        self.final_tokens = 260e9
        self.ckpt_path = 'modelv10.pt'
        self.num_workers = 0
        for k, v in kwargs.items():
            setattr(self, k, v)

def train(model, loader, optimizer, config, device='cuda', guidance=True):
    model.train()
    scaler = GradScaler()
    tokens_processed = 0
    start_time = time.time()

    latent_template = torch.zeros(config.batch_size, config.latent_dim, device=device)

    for epoch in range(config.max_epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{config.max_epochs}", ncols=120)

        for batch_idx, (x, props) in enumerate(pbar):
            x, props = x.to(device, non_blocking=True), props.to(device, non_blocking=True)
            latent = latent_template[:x.size(0)]

            inpt, outpt = x[:, :-1], x[:, 1:]

            with autocast():
                if guidance:
                    mask = (torch.rand(x.size(0), 1, device=device) < 0.1).float()
                    props_masked = props * (1 - mask)
                    latent_masked = latent * (1 - mask)
                else:
                    props_masked, latent_masked = props, latent

                logits, loss = model(inpt, props_masked, targets=outpt, latent=latent_masked)
                loss = loss.mean()

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            scaler.step(optimizer)
            scaler.update()

            if config.lr_decay:
                tokens_processed += (outpt >= 0).sum().item()
                if tokens_processed < config.warmup_tokens:
                    lr_mult = float(tokens_processed) / float(max(1, config.warmup_tokens))
                else:
                    progress = (tokens_processed - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                    lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                lr = config.learning_rate * lr_mult
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
            else:
                lr = config.learning_rate

            if batch_idx % 20 == 0:
                pbar.set_postfix(Loss=f"{loss.item():.4f}", LR=f"{lr:.6f}")

            if batch_idx % 200 == 0:
                torch.cuda.empty_cache()

            del loss, logits, inpt, outpt

        torch.save(model.state_dict(), f"{config.ckpt_path}_epoch_{epoch+1}.pt")

    total_time = time.time() - start_time
    print(f"✅ Training finished in {total_time/3600:.2f} hours.")

train_config = TrainerConfig(batch_size=128)

optimizer = modelv10.configure_optimizers(train_config)
train(modelv10, loader, optimizer, train_config, device='cuda', guidance=True)





