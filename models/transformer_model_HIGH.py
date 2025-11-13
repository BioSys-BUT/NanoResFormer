#!/home/vorochta/miniconda3/envs/tfenv/bin/python
# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn

#########################################
# Konvoluční feature extractor
#########################################

class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=3,
                               stride=1, padding=1, padding_mode='replicate')
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=out_ch, out_channels=out_ch, kernel_size=3,
                               stride=1, padding=1, padding_mode='replicate')
        self.BN = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.BN(x)
        return x


class Encoder(nn.Module):
    def __init__(self, chs=(1, 16, 32, 64, 128)):      #změna modelu
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        for block in self.enc_blocks:
            x = block(x)
            x = self.pool(x)
        return x


class FeatureExtractor(nn.Module):
    def __init__(self, enc_chs=(1, 16, 32, 64, 128)):      #změna modelu
        super().__init__()
        self.encoder = Encoder(enc_chs)

    def forward(self, x):
        x = x.unsqueeze(1)
        x_enc = self.encoder(x)
        return x_enc.transpose(1, 2)  # (B, L_enc, C_out)


#########################################
# Transformer část
#########################################

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=10000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ff(self.norm2(x))
        return x


#########################################
# Kompletní model: Feature extractor + Transformer
#########################################

class FullModel(nn.Module):
    def __init__(self, segment_length=20000, d_model=64, n_heads=8, dropout=0.1,
                 n_transformer_layers=4, num_classes=2, pretrained_path=None):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.input_proj = nn.Linear(128, d_model) #změna modelu
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=5000)
        self.transformer_layers = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, dropout) for _ in range(n_transformer_layers)]
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, num_classes)

        if pretrained_path is not None:
            self.load_state_dict(torch.load(pretrained_path))
            print("Načteny předtrénované váhy.")

    def forward(self, x):
        features = self.feature_extractor(x)
        x_proj = self.input_proj(features)
        x_enc = self.pos_encoder(x_proj)
        for layer in self.transformer_layers:
            x_enc = layer(x_enc)
        x_transposed = x_enc.transpose(1, 2)
        pooled = self.global_pool(x_transposed).squeeze(-1)
        logits = self.classifier(pooled)
        return logits


# if __name__ == "__main__":
#     dummy_input = torch.randn(8, 20000)
#     model = FullModel()
#     output = model(dummy_input)
#     print("Tvar výstupu:", output.shape)
