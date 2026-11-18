"""Conv1D + small Transformer over a 30-frame keypoint window -> 7 word logits (about 1.3 M parameters).

Attention is written out with matmul/softmax (not nn.MultiheadAttention) so the ONNX export and the TensorRT parser see
only plain ops. Input [B, 30, 233] features from signfeat.py, output [B, 7] logits (softmax is applied by the server).
"""
import torch
from torch import nn

import signfeat as S


class Block(nn.Module):
    def __init__(self, d, heads, ff, p):
        super().__init__()
        self.h, self.d = heads, d
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(p), nn.Linear(ff, d))
        self.drop = nn.Dropout(p)

    def forward(self, x):
        B, T, _ = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, self.d // self.h).permute(2, 0, 3, 1, 4)
        att = torch.softmax(q @ k.transpose(-1, -2) / (self.d // self.h) ** 0.5, dim=-1)
        x = x + self.drop(self.proj((att @ v).transpose(1, 2).reshape(B, T, self.d)))
        return x + self.drop(self.ff(self.n2(x)))


class SignNet(nn.Module):
    def __init__(self, n_cls=7, f_in=S.F_DIM, d=192, layers=3, heads=4, ff=384, p=0.15):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(f_in, d), nn.LayerNorm(d))
        self.conv = nn.ModuleList([nn.Sequential(nn.Conv1d(d, d, 5, padding=2), nn.BatchNorm1d(d), nn.GELU()) for _ in range(2)])
        self.pos = nn.Parameter(torch.zeros(1, S.WINDOW, d))
        self.blocks = nn.ModuleList([Block(d, heads, ff, p) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(2 * d, n_cls))

    def forward(self, x):
        x = self.inp(x)
        for c in self.conv:  # residual local-motion layers
            x = x + c(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.pos
        for b in self.blocks:
            x = b(x)
        x = self.norm(x)
        return self.head(torch.cat([x.mean(1), x.max(1).values], dim=-1))
