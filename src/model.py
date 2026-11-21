"""
The neural network that predicts next month's spending per category.

Design (kept intentionally simple):

  1. Scale-free inputs. A $50 change means something different for someone spending
     $1,500/month than for someone spending $8,000/month, so every input is divided by
     the person's average monthly spending ("scale"). The model learns patterns in
     *proportions*, and the scale itself is given as one extra feature.

  2. Baseline + correction. The output is
         prediction = history average + learned correction
     and the last layer starts at zero, so before training the network behaves exactly
     like the full-history-average baseline. Training only has to learn *how to adjust*
     the average (seasonality, trends, income changes), which works well with little data.

  3. A small multi-layer perceptron (MLP): Linear -> ReLU -> Dropout, repeated.
"""

import math

import torch
from torch import nn


class SpendingModel(nn.Module):
    def __init__(self, history: int, n_features: int, n_categories: int,
                 hidden: int = 128, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.history = history
        self.n_features = n_features
        self.n_categories = n_categories

        in_dim = history * n_features + 3          # + log(scale), sin(month), cos(month)
        blocks, d = [], in_dim
        for _ in range(layers):
            blocks += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        self.body = nn.Sequential(*blocks)
        self.head = nn.Linear(d, n_categories)
        nn.init.zeros_(self.head.weight)           # start out == baseline
        nn.init.zeros_(self.head.bias)

    def scale(self, hist: torch.Tensor) -> torch.Tensor:
        """Average total monthly spending over the history window, shape (N,)."""
        return hist[:, :, :self.n_categories].sum(dim=2).mean(dim=1) + 1.0

    def forward(self, hist: torch.Tensor, month: torch.Tensor) -> torch.Tensor:
        """
        hist:  (N, H, F) past months in dollars ([categories..., income])
        month: (N,) calendar month 1-12 of the month being predicted
        returns (N, C) predicted dollars (may be slightly negative during training;
        use .predict() for final, non-negative predictions)
        """
        s = self.scale(hist)                                   # (N,)
        x = (hist / s[:, None, None]).clamp(max=10.0)          # proportions, cap spikes
        angle = 2 * math.pi * (month.float() - 1) / 12
        extra = torch.stack([torch.log(s) / 10, torch.sin(angle), torch.cos(angle)], dim=1)
        features = torch.cat([x.flatten(1), extra], dim=1)

        baseline = x[:, :, :self.n_categories].mean(dim=1)     # history average (scaled)
        correction = self.head(self.body(features))
        return (baseline + correction) * s[:, None]

    @torch.no_grad()
    def predict(self, hist, month) -> torch.Tensor:
        self.eval()
        return self.forward(hist, month).clamp(min=0.0)
