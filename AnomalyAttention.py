
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class AnomalyAttention(nn.Module):
    def __init__(self, N, d_model, heads):
        super(AnomalyAttention, self).__init__()
        self.N = N
        self.d_model = d_model
        self.heads = heads
        self.Wq = nn.ModuleList(nn.Linear(d_model, d_model // heads, bias=False) for _ in range(self.heads))
        self.Wk = nn.ModuleList(nn.Linear(d_model, d_model // heads, bias=False) for _ in range(self.heads))
        self.Wv = nn.ModuleList(nn.Linear(d_model, d_model // heads, bias=False) for _ in range(self.heads))
        self.Ws = nn.ModuleList(nn.Linear(d_model, 1, bias=False) for _ in range(self.heads))
        self.w0 = nn.Linear(d_model, d_model)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def attention(self, Q, K, V):
        scores = Q @ K.transpose(-2, -1)
        scaling_factor = math.sqrt(Q.size(-1))
        scores = scores / scaling_factor
        scores = (scores - torch.mean(scores, dim=-1)) / torch.std(scores, dim=-1)
        weights = F.softmax(scores, dim=-1)
        Z = weights @ V
        return Z, weights

    def forward(self, x):
        head_outputs = []
        prior_asso = []
        series_asso = []
        for i in range(self.heads):
            Q = self.Wq[i](x)
            K = self.Wk[i](x)
            V = self.Wv[i](x)
            Z, S = self.attention(Q, K, V)
            sigma = torch.clamp(self.Ws[i](x), min=1e-3)
            head_outputs.append(Z)
            series_asso.append(S)
            P = self.prior_association(sigma, self.device)
            prior_asso.append(P)
        concatenated_outputs = torch.cat(head_outputs, dim=-1)
        Z = self.w0(concatenated_outputs)
        return Z, prior_asso, series_asso

    @staticmethod
    def prior_association(sigma, device):
        N = sigma.shape[0]
        p = torch.from_numpy(np.abs(np.indices((N, N))[0] - np.indices((N, N))[1])).to(device)
        gaussian = torch.exp(-0.5 * (p / sigma).pow(2)) / torch.sqrt(2 * torch.pi * sigma)
        prior_ass = gaussian / gaussian.sum(axis=1, keepdim=True)
        return prior_ass
