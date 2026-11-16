"""Published spatio-temporal forecasting baselines, re-implemented on the shared Forecaster interface.

AGCRN        Bai et al., NeurIPS 2020  - node-adaptive graph convolutional recurrent network
GraphWaveNet Wu et al., IJCAI 2019     - gated dilated TCN + diffusion GCN with adaptive adjacency
STAEformer   Liu et al., CIKM 2023     - vanilla transformer + spatio-temporal adaptive embedding

Inputs follow DemandDataset: x [B, P, N, 2] (3 same-slot-previous-day steps, then 12 recent),
x_tod/x_dow [B, P]. Defaults follow the official repositories, scaled to the 15-step input.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .forecasting import Forecaster


# ----------------------------------------------------------------------------- AGCRN
class AVWGCN(nn.Module):
    """Adaptive graph conv with node-adaptive parameters: W_n = E_n · W_pool (NAPL)."""

    def __init__(self, c_in, c_out, cheb_k, embed_dim):
        super().__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.randn(embed_dim, cheb_k, c_in, c_out) * (1.0 / c_in) ** 0.5)
        self.bias_pool = nn.Parameter(torch.zeros(embed_dim, c_out))

    def forward(self, x, emb):                                   # x [B, N, C], emb [N, d]
        n = emb.shape[0]
        a = F.softmax(F.relu(emb @ emb.T), dim=1)                # DAGG adaptive adjacency
        supports = [torch.eye(n, device=x.device), a]
        for _ in range(2, self.cheb_k):
            supports.append(2 * a @ supports[-1] - supports[-2])
        s = torch.stack(supports)                                # [K, N, N]
        w = torch.einsum("nd,dkio->nkio", emb, self.weights_pool)
        b = emb @ self.bias_pool
        xg = torch.einsum("knm,bmc->bnkc", s, x)
        return torch.einsum("bnki,nkio->bno", xg, w) + b


class AGCRNCell(nn.Module):
    def __init__(self, c_in, hidden, cheb_k, embed_dim):
        super().__init__()
        self.hidden = hidden
        self.gate = AVWGCN(c_in + hidden, 2 * hidden, cheb_k, embed_dim)
        self.update = AVWGCN(c_in + hidden, hidden, cheb_k, embed_dim)

    def forward(self, x, state, emb):
        zr = torch.sigmoid(self.gate(torch.cat([x, state], -1), emb))
        z, r = zr.split(self.hidden, dim=-1)
        hc = torch.tanh(self.update(torch.cat([x, z * state], -1), emb))
        return r * state + (1 - r) * hc


class AGCRN(Forecaster):
    def __init__(self, scaler, num_nodes, horizon, hidden=64, embed_dim=10, cheb_k=2, layers=2, in_dim=2):
        super().__init__(scaler)
        self.horizon, self.in_dim = horizon, in_dim
        self.emb = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.cells = nn.ModuleList([AGCRNCell(in_dim if i == 0 else hidden, hidden, cheb_k, embed_dim)
                                    for i in range(layers)])
        self.end_conv = nn.Conv2d(1, horizon * in_dim, kernel_size=(1, hidden))

    def forward_scaled(self, batch):
        x = batch["x"]                                           # [B, P, N, C]
        b, p, n, _ = x.shape
        seq = x
        for cell in self.cells:
            h = torch.zeros(b, n, cell.hidden, device=x.device)
            outs = []
            for t in range(p):
                h = cell(seq[:, t], h, self.emb)
                outs.append(h)
            seq = torch.stack(outs, 1)
        out = self.end_conv(seq[:, -1:])                         # [B, H*C, N, 1]
        return out.squeeze(-1).reshape(b, self.horizon, self.in_dim, n).permute(0, 1, 3, 2)


# ----------------------------------------------------------------------------- Graph WaveNet
class DiffusionGCN(nn.Module):
    def __init__(self, c_in, c_out, n_supports, order=2, dropout=0.3):
        super().__init__()
        self.order, self.dropout = order, dropout
        self.mlp = nn.Conv2d(c_in * (n_supports * order + 1), c_out, kernel_size=1)

    def forward(self, x, supports):                              # x [B, C, N, T]
        out = [x]
        for a in supports:
            x1 = torch.einsum("bcnt,nm->bcmt", x, a)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x1 = torch.einsum("bcnt,nm->bcmt", x1, a)
                out.append(x1)
        return F.dropout(self.mlp(torch.cat(out, 1)), self.dropout, self.training)


class GraphWaveNet(Forecaster):
    def __init__(self, scaler, num_nodes, horizon, adj: torch.Tensor, in_dim=2, residual=32, dilation=32,
                 skip=256, end=512, blocks=4, layers=2, kernel=2, apt_dim=10, dropout=0.3):
        super().__init__(scaler)
        self.horizon, self.in_dim = horizon, in_dim
        adj = adj + torch.eye(num_nodes)
        self.register_buffer("p_fwd", adj / adj.sum(1, keepdim=True))
        self.register_buffer("p_bwd", adj.T / adj.T.sum(1, keepdim=True))
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, apt_dim))
        self.nodevec2 = nn.Parameter(torch.randn(apt_dim, num_nodes))
        self.start = nn.Conv2d(in_dim, residual, 1)
        self.filters, self.gates, self.skips, self.gconvs, self.bns = (nn.ModuleList() for _ in range(5))
        self.receptive_field = 1
        for _ in range(blocks):
            d, extra = 1, kernel - 1
            for _ in range(layers):
                self.filters.append(nn.Conv2d(residual, dilation, (1, kernel), dilation=d))
                self.gates.append(nn.Conv2d(residual, dilation, (1, kernel), dilation=d))
                self.skips.append(nn.Conv2d(dilation, skip, 1))
                self.gconvs.append(DiffusionGCN(dilation, residual, n_supports=3, dropout=dropout))
                self.bns.append(nn.BatchNorm2d(residual))
                self.receptive_field += extra
                extra *= 2
                d *= 2
        self.end1 = nn.Conv2d(skip, end, 1)
        self.end2 = nn.Conv2d(end, horizon * in_dim, 1)

    def forward_scaled(self, batch):
        x = batch["x"].permute(0, 3, 2, 1)                       # [B, C, N, T]
        if x.shape[-1] < self.receptive_field:
            x = F.pad(x, (self.receptive_field - x.shape[-1], 0))
        adp = F.softmax(F.relu(self.nodevec1 @ self.nodevec2), dim=1)
        supports = [self.p_fwd, self.p_bwd, adp]
        x = self.start(x)
        skip = 0
        for f, g, s, gc, bn in zip(self.filters, self.gates, self.skips, self.gconvs, self.bns):
            res = x
            x = torch.tanh(f(x)) * torch.sigmoid(g(x))
            sk = s(x)
            skip = sk + (skip[..., -sk.shape[-1]:] if torch.is_tensor(skip) else 0)
            x = gc(x, supports) + res[..., -x.shape[-1]:]
            x = bn(x)
        out = self.end2(F.relu(self.end1(F.relu(skip[..., -1:]))))   # [B, H*C, N, 1]
        b, _, n, _ = out.shape
        return out.squeeze(-1).reshape(b, self.horizon, self.in_dim, n).permute(0, 1, 3, 2)


# ----------------------------------------------------------------------------- STAEformer
class _AxisAttention(nn.Module):
    def __init__(self, dim, heads, ff, dropout):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(dim, heads, ff, dropout, batch_first=True)

    def forward(self, x, axis):                                  # x [B, T, N, D]; attend along axis
        b, t, n, d = x.shape
        if axis == 1:
            y = self.layer(x.permute(0, 2, 1, 3).reshape(b * n, t, d))
            return y.reshape(b, n, t, d).permute(0, 2, 1, 3)
        return self.layer(x.reshape(b * t, n, d)).reshape(b, t, n, d)


class STAEformer(Forecaster):
    def __init__(self, scaler, num_nodes, horizon, in_steps, slots_per_day, in_dim=2, input_emb=24, tod_emb=24,
                 dow_emb=24, adaptive_emb=80, heads=4, ff=256, layers=3, dropout=0.1):
        super().__init__(scaler)
        self.horizon, self.in_dim, self.in_steps = horizon, in_dim, in_steps
        self.dim = input_emb + tod_emb + dow_emb + adaptive_emb
        self.inp = nn.Linear(in_dim, input_emb)
        self.tod = nn.Embedding(slots_per_day, tod_emb)
        self.dow = nn.Embedding(7, dow_emb)
        self.adaptive = nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_emb))
        nn.init.xavier_uniform_(self.adaptive)
        self.temporal = nn.ModuleList([_AxisAttention(self.dim, heads, ff, dropout) for _ in range(layers)])
        self.spatial = nn.ModuleList([_AxisAttention(self.dim, heads, ff, dropout) for _ in range(layers)])
        self.out = nn.Linear(in_steps * self.dim, horizon * in_dim)

    def forward_scaled(self, batch):
        x = batch["x"]
        b, t, n, _ = x.shape
        h = torch.cat([self.inp(x),
                       self.tod(batch["x_tod"])[:, :, None].expand(b, t, n, -1),
                       self.dow(batch["x_dow"])[:, :, None].expand(b, t, n, -1),
                       self.adaptive.expand(b, t, n, -1)], dim=-1)
        for layer in self.temporal:
            h = layer(h, axis=1)
        for layer in self.spatial:
            h = layer(h, axis=2)
        out = self.out(h.permute(0, 2, 1, 3).reshape(b, n, t * self.dim))
        return out.reshape(b, n, self.horizon, self.in_dim).permute(0, 2, 1, 3)
