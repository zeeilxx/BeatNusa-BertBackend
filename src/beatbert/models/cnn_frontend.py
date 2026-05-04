from __future__ import annotations

import torch
from torch import nn


class CNNFrontend(nn.Module):
    def __init__(self, input_mels: int, channels: list[int], d_model: int, dropout: float = 0.1):
        super().__init__()
        c1, c2, c3 = channels
        self.net = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.GELU(),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm2d(c3),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Dropout(dropout),
        )
        reduced_mels = input_mels // 4
        self.proj = nn.Linear(c3 * reduced_mels, d_model)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: [B, M, T]
        x = mel.unsqueeze(1)
        x = self.net(x)  # [B, C, M', T]
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, T, C, M']
        x = x.flatten(start_dim=2)  # [B, T, C*M']
        return self.proj(x)
