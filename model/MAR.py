## tkx to lucidrains for the code


from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        if embed_size % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, embed_size)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, embed_size)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MAR(nn.Module):
    def __init__(self, embed_size, nhead, num_layers, dropout=0.1):
        super().__init__()
        self.embed_size = embed_size

        self.modality_embed = nn.Embedding(4, embed_size)

        self.pos_encoder = PositionalEncoding(embed_size, dropout)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(embed_size, nhead, dropout=dropout)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)

        self.out_proj = nn.Linear(embed_size, embed_size)

        self.mi_estimator = nn.Sequential(
            nn.Linear(2 * embed_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.loss_fn = nn.MSELoss()

    def interleave(self, a, b, c, d):
        max_len = max(a.size(1), b.size(1), c.size(1), d.size(1))

        a = a.expand(-1, max_len, -1)
        b = b.expand(-1, max_len, -1)
        c = c.expand(-1, max_len, -1)
        d = d.expand(-1, max_len, -1)

        combined = torch.stack([a, b, c, d], dim=2)  # [B, max_len, 4, D]
        return combined.view(a.size(0), -1, self.embed_size)  # [B, 4 * max_len, D]

    def forward(self, a, b, c, d):
        # Interleave modalities
        interleaved = self.interleave(a, b, c, d)  # [B, 4 * max_len, D]

        # Add SOS token
        sos = torch.zeros_like(interleaved[:, :1])  # [B, 1, D]
        x = torch.cat([sos, interleaved], dim=1)  # [B, 4 * max_len + 1, D]

        # Generate modality IDs
        modality_ids = torch.zeros(x.size(1), dtype=torch.long, device=x.device)
        modality_ids[1::4] = 0  # a
        modality_ids[2::4] = 1  # b
        modality_ids[3::4] = 2  # c
        modality_ids[4::4] = 3  # d
        modality_ids = modality_ids.unsqueeze(0).expand(x.size(0), -1)  # [B, T]

        # Combine embeddings
        x = x + self.modality_embed(modality_ids)
        x = self.pos_encoder(x)

        # Auto-regressive processing
        x = x.transpose(0, 1)  # [T, B, D]
        mask = torch.triu(torch.ones(x.size(0), x.size(0)) * float('-inf'), diagonal=1).to(x.device)
        out = self.transformer(tgt=x, memory=x, tgt_mask=mask)  # [T, B, D]
        out = out.transpose(0, 1)  # [B, T, D]

        # Reconstruction loss
        pred = self.out_proj(out[:, :-1])  # Predict sequence (exclude SOS)
        target = interleaved
        recon_loss = self.loss_fn(pred, target)

        # Mutual information regularization
        a_feats = out[:, 1::4, :]
        b_feats = out[:, 2::4, :]
        c_feats = out[:, 3::4, :]
        d_feats = out[:, 4::4, :]

        # Positive pairs
        pos_pairs = [
            torch.cat([a_feats, b_feats], dim=-1),  # (a, b)
            torch.cat([a_feats, c_feats], dim=-1),  # (a, c)
            torch.cat([a_feats, d_feats], dim=-1),  # (a, d)
            torch.cat([b_feats, c_feats], dim=-1),  # (b, c)
            torch.cat([b_feats, d_feats], dim=-1),  # (b, d)
            torch.cat([c_feats, d_feats], dim=-1),  # (c, d)
        ]

        # Negative pairs (shuffle one modality in each pair)
        neg_pairs = [
            torch.cat([a_feats, b_feats[:, torch.randperm(b_feats.size(1))]], dim=-1),  # Shuffle b
            torch.cat([a_feats, c_feats[:, torch.randperm(c_feats.size(1))]], dim=-1),  # Shuffle c
            torch.cat([a_feats, d_feats[:, torch.randperm(d_feats.size(1))]], dim=-1),  # Shuffle d
            torch.cat([b_feats, c_feats[:, torch.randperm(c_feats.size(1))]], dim=-1),  # Shuffle c
            torch.cat([b_feats, d_feats[:, torch.randperm(d_feats.size(1))]], dim=-1),  # Shuffle d
            torch.cat([c_feats, d_feats[:, torch.randperm(d_feats.size(1))]], dim=-1),  # Shuffle d
        ]

        # Compute MI loss for each pair
        mi_loss = 0
        for pos, neg in zip(pos_pairs, neg_pairs):
            mi_logits = self.mi_estimator(pos) - self.mi_estimator(neg)
            mi_loss += -F.logsigmoid(mi_logits).mean()

        total_loss = recon_loss +   mi_loss
        return out[:, 1:, :], total_loss