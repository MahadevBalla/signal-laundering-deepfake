"""Neural backends used on top of frozen SSL features.

This module defines the classifier heads used during training and evaluation:
- FFN backend
- AASIST-style graph backend
- RawNet2-inspired GRU backend
- weighted layer aggregation wrappers
"""

from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_aasist_path = os.path.join(os.path.dirname(__file__), "..", "..", "external", "aasist", "models")
if _aasist_path not in sys.path:
    sys.path.insert(0, _aasist_path)

try:
    from AASIST import GraphAttentionLayer, HtrgGraphAttentionLayer, GraphPool
    _AASIST_AVAILABLE = True
except ImportError:
    _AASIST_AVAILABLE = False


class AttentiveStatisticalPooling(nn.Module):
    """Attentive pooling that returns weighted mean and standard deviation."""

    def __init__(self, in_dim: int):
        """Create the attention block for temporal pooling."""
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Tanh(),
            nn.Linear(in_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool a sequence tensor `[B, T, D]` into a fixed `[B, 2D]` vector."""
        w = torch.softmax(self.attn(x), dim=1)
        mean = (w * x).sum(dim=1)
        std = ((w * (x - mean.unsqueeze(1)) ** 2).sum(dim=1) + 1e-9).sqrt()
        return torch.cat([mean, std], dim=-1)


class FFNBackend(nn.Module):
    """Feed-forward backend used for SSL spoof classification."""

    def __init__(self, embed_dim: int = 768, hidden: int = 128, dropout: float = 0.2):
        """Initialize FFN layers, attentive pooling, and classifier head."""
        super().__init__()
        self.bn = nn.BatchNorm1d(embed_dim)
        self.ff1 = nn.Linear(embed_dim, hidden)
        self.ff2 = nn.Linear(hidden, hidden)
        self.act = nn.SELU(inplace=False)
        self.drop = nn.Dropout(p=dropout, inplace=False)
        self.pool = AttentiveStatisticalPooling(hidden)
        self.head = nn.Linear(2 * hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run temporal FFN encoding and return 2-class logits."""
        B, T, D = x.shape
        x = self.bn(x.reshape(B * T, D)).reshape(B, T, D)
        x = self.drop(self.act(self.ff1(x)))
        x = self.drop(self.act(self.ff2(x)))
        x = self.pool(x)
        return self.head(x)


class _SSLResidualBlock(nn.Module):
    """Residual 2D block used by the SSL-adapted AASIST backend."""

    def __init__(self, nb_filts, first: bool = False):
        """Build one residual unit with optional channel projection."""
        super().__init__()
        self.first = first
        if not self.first:
            self.bn1 = nn.BatchNorm2d(nb_filts[0])
        self.conv1 = nn.Conv2d(nb_filts[0], nb_filts[1], kernel_size=(2, 3), padding=(1, 1), stride=1)
        self.selu = nn.SELU(inplace=False)
        self.bn2 = nn.BatchNorm2d(nb_filts[1])
        self.conv2 = nn.Conv2d(nb_filts[1], nb_filts[1], kernel_size=(2, 3), padding=(0, 1), stride=1)
        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(nb_filts[0], nb_filts[1], padding=(0, 1), kernel_size=(1, 3), stride=1)
        else:
            self.downsample = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual convolutional block on SSL feature maps."""
        identity = x
        out = x if self.first else self.selu(self.bn1(x))
        out = self.selu(self.bn2(self.conv1(x)))
        out = self.conv2(out)
        if self.downsample:
            identity = self.conv_downsample(identity)
        return out + identity


class AASISTBackend(nn.Module):
    """AASIST-style graph-attention backend adapted for SSL features."""

    def __init__(self, embed_dim: int = 768, proj_dim: int = 128):
        """Construct the full AASIST-style stack after SSL projection."""
        super().__init__()
        if not _AASIST_AVAILABLE:
            raise ImportError(
                "Cannot import from external/aasist. Run: git submodule update --init external/aasist"
            )
        filts = [proj_dim, [1, 32], [32, 32], [32, 64], [64, 64]]
        gat_dims = [64, 32]
        pool_ratios = [0.5, 0.5, 0.5, 0.5]
        temperatures = [2.0, 2.0, 100.0, 100.0]
        spec_nodes = math.floor((proj_dim - 3) / 3) + 1

        self.LL = nn.Linear(embed_dim, proj_dim)
        self.first_bn = nn.BatchNorm2d(1)
        self.first_bn1 = nn.BatchNorm2d(64)
        self.drop = nn.Dropout(0.5, inplace=False)
        self.drop_way = nn.Dropout(0.2, inplace=False)
        self.selu = nn.SELU(inplace=False)
        self.encoder = nn.Sequential(
            _SSLResidualBlock(filts[1], first=True),
            _SSLResidualBlock(filts[2]),
            _SSLResidualBlock(filts[3]),
            _SSLResidualBlock(filts[4]),
            _SSLResidualBlock(filts[4]),
            _SSLResidualBlock(filts[4]),
        )
        self.attention = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1)),
            nn.SELU(inplace=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=(1, 1)),
        )
        self.pos_S = nn.Parameter(torch.randn(1, spec_nodes, gat_dims[0]))
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.GAT_layer_S = GraphAttentionLayer(64, gat_dims[0], temperature=temperatures[0])
        self.GAT_layer_T = GraphAttentionLayer(64, gat_dims[0], temperature=temperatures[1])
        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(gat_dims[1], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(gat_dims[0], gat_dims[1], temperature=temperatures[3])
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(gat_dims[1], gat_dims[1], temperature=temperatures[3])
        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.out_layer = nn.Linear(5 * gat_dims[1], 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the SSL-to-AASIST graph pipeline and return logits."""
        x = self.LL(x)
        x = x.transpose(1, 2).unsqueeze(1)
        x = self.selu(self.first_bn(F.max_pool2d(x, (3, 3))))
        x = self.selu(self.first_bn1(self.encoder(x)))
        w = self.attention(x)
        e_S = torch.sum(x * F.softmax(w, dim=-1), dim=-1).transpose(1, 2) + self.pos_S
        e_T = torch.sum(x * F.softmax(w, dim=-2), dim=-2).transpose(1, 2)
        out_S = self.pool_S(self.GAT_layer_S(e_S))
        out_T = self.pool_T(self.GAT_layer_T(e_T))
        m1 = self.master1.expand(x.size(0), -1, -1)
        m2 = self.master2.expand(x.size(0), -1, -1)
        out_T1, out_S1, m1 = self.HtrgGAT_layer_ST11(out_T, out_S, master=m1)
        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)
        dT, dS, dm = self.HtrgGAT_layer_ST12(out_T1, out_S1, master=m1)
        out_T1 = out_T1 + dT
        out_S1 = out_S1 + dS
        m1 = m1 + dm
        out_T2, out_S2, m2 = self.HtrgGAT_layer_ST21(out_T, out_S, master=m2)
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)
        dT, dS, dm = self.HtrgGAT_layer_ST22(out_T2, out_S2, master=m2)
        out_T2 = out_T2 + dT
        out_S2 = out_S2 + dS
        m2 = m2 + dm
        out_T1 = self.drop_way(out_T1); out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1); out_S2 = self.drop_way(out_S2)
        m1 = self.drop_way(m1); m2 = self.drop_way(m2)
        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(m1, m2)
        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)
        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)
        hidden = torch.cat([T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)
        return self.out_layer(self.drop(hidden))


class RawNet2Backend(nn.Module):
    """RawNet2-inspired recurrent backend for SSL embeddings."""

    def __init__(self, embed_dim: int = 768, hidden: int = 128, gru_hidden: int = 1024, gru_layers: int = 3, dropout: float = 0.2):
        """Initialize projection, GRU stack, and classifier head."""
        super().__init__()
        self.bn = nn.BatchNorm1d(embed_dim)
        self.project = nn.Linear(embed_dim, hidden)
        self.act = nn.LeakyReLU(0.3, inplace=True)
        self.gru = nn.GRU(
            input_size=hidden,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(gru_hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode SSL sequence with GRU and predict spoof logits."""
        B, T, D = x.shape
        x = self.bn(x.reshape(B * T, D)).reshape(B, T, D)
        x = self.act(self.project(x))
        x, _ = self.gru(x)
        return self.fc(self.drop(x[:, -1, :]))


def _weighted_sum(layer_states: dict, weights: nn.Parameter) -> torch.Tensor:
    """Compute softmax-weighted sum over layer-wise SSL states."""
    layers = sorted(layer_states)
    stack = torch.stack([layer_states[l] for l in layers], dim=-1)
    w = torch.softmax(weights, dim=0)
    return (stack * w).sum(dim=-1)


def _get_layer_weights(weights: nn.Parameter) -> dict[int, float]:
    """Return normalized layer weights as a plain Python dict."""
    w = torch.softmax(weights.detach().cpu(), dim=0)
    return {i: float(w[i]) for i in range(len(w))}


class WeightedAggregationBackend(nn.Module):
    """Weighted SSL aggregation followed by the FFN backend."""

    def __init__(self, num_layers: int = 12, embed_dim: int = 768, hidden: int = 128, dropout: float = 0.2):
        """Initialize learnable layer weights and FFN head."""
        super().__init__()
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))
        self.ffn = FFNBackend(embed_dim=embed_dim, hidden=hidden, dropout=dropout)

    def forward(self, layer_states: dict) -> torch.Tensor:
        """Aggregate SSL layers and return FFN logits."""
        return self.ffn(_weighted_sum(layer_states, self.layer_weights))

    def get_layer_weights(self) -> dict[int, float]:
        """Expose learned layer importances after softmax normalization."""
        return _get_layer_weights(self.layer_weights)


class SSLWithAASIST(nn.Module):
    """Weighted SSL aggregation followed by the AASIST backend."""

    def __init__(self, num_layers: int = 12, embed_dim: int = 768, proj_dim: int = 128):
        """Initialize learnable layer weights and AASIST backend."""
        super().__init__()
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))
        self.backend = AASISTBackend(embed_dim=embed_dim, proj_dim=proj_dim)

    def forward(self, layer_states: dict) -> torch.Tensor:
        """Aggregate SSL layers and pass them to the AASIST backend."""
        return self.backend(_weighted_sum(layer_states, self.layer_weights))

    def get_layer_weights(self) -> dict[int, float]:
        """Return normalized SSL layer weights for inspection."""
        return _get_layer_weights(self.layer_weights)


class SSLWithRawNet2(nn.Module):
    """Weighted SSL aggregation followed by the RawNet2-style backend."""

    def __init__(self, num_layers: int = 12, embed_dim: int = 768, hidden: int = 128, gru_hidden: int = 1024):
        """Initialize learnable layer weights and recurrent backend."""
        super().__init__()
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))
        self.backend = RawNet2Backend(embed_dim=embed_dim, hidden=hidden, gru_hidden=gru_hidden)

    def forward(self, layer_states: dict) -> torch.Tensor:
        """Aggregate SSL layers and pass them to the GRU backend."""
        return self.backend(_weighted_sum(layer_states, self.layer_weights))

    def get_layer_weights(self) -> dict[int, float]:
        """Return normalized SSL layer weights for reporting."""
        return _get_layer_weights(self.layer_weights)
