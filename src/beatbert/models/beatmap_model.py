from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from beatbert.models.cnn_frontend import CNNFrontend
from beatbert.models.transformer import BeatTransformerEncoder, PositionalEmbedding


class BeatmapModel(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        mcfg = cfg['model']
        self.frontend = CNNFrontend(
            input_mels=mcfg['input_mels'],
            channels=mcfg['cnn_channels'],
            d_model=mcfg['d_model'],
            dropout=mcfg['dropout'],
        )
        self.positional = PositionalEmbedding(mcfg['max_seq_len'], mcfg['d_model'])
        self.encoder = BeatTransformerEncoder(
            d_model=mcfg['d_model'],
            num_layers=mcfg['num_layers'],
            num_heads=mcfg['num_heads'],
            ff_mult=mcfg['ff_mult'],
            dropout=mcfg['dropout'],
        )
        self.norm = nn.LayerNorm(mcfg['d_model'])
        self.event_head = nn.Linear(mcfg['d_model'], 1)
        self.lane_head = nn.Linear(mcfg['d_model'], mcfg['num_lanes'])
        self.predict_note_type = bool(mcfg.get('predict_note_type', True))
        self.type_head = nn.Linear(mcfg['d_model'], 2) if self.predict_note_type else None

    def forward(self, mel: torch.Tensor, attention_mask: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        x = self.frontend(mel)
        x = self.positional(x)
        x = self.encoder(x, attention_mask=attention_mask)
        x = self.norm(x)
        out = {
            'event_logits': self.event_head(x).squeeze(-1),
            'lane_logits': self.lane_head(x),
        }
        if self.predict_note_type:
            out['type_logits'] = self.type_head(x)
        return out
