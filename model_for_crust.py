# model.py
import torch
import torch.nn as nn
import math

class TinyGPTBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_mult=4, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Linear(ff_mult * d_model, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # attn_mask is (T, T) boolean where True=masked (future)
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = self.ln1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.ln2(x + self.dropout(ff_out))
        return x


class TinyGPTClassifier(nn.Module):
    """
    Tiny GPT-style decoder turned into a classifier.
    Exposes:
      - forward(input_ids, attention_mask) -> logits (B, C)
      - encode(input_ids, attention_mask) -> features (B, D) (pooled last token)
    """

    def __init__(self, vocab_size, max_len=64, d_model=128, n_layers=4, n_heads=4, num_classes=2, dropout=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.blocks = nn.ModuleList([
            TinyGPTBlock(d_model, n_heads, ff_mult=4, dropout=dropout) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)

        # classification head
        self.classifier = nn.Linear(d_model, num_classes)

        # tie weights is optional; for classifier not tying usually
        # self.classifier.weight = self.token_emb.weight[:num_classes]  # not recommended

    def _make_causal_mask(self, T, device):
        # PyTorch MultiheadAttention expects attn_mask shape (T, T) with True at positions to be masked
        return torch.triu(torch.ones((T, T), dtype=torch.bool, device=device), diagonal=1)

    def encode(self, input_ids, attention_mask=None):
        """
        Returns penultimate pooled features (B, d_model)
        Use this to compute last-layer gradient embedding without backprop.
        """
        B, T = input_ids.shape
        device = input_ids.device
        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        x = self.token_emb(input_ids) + self.pos_emb(pos)

        causal_mask = self._make_causal_mask(T, device)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0  # True for pads

        for blk in self.blocks:
            x = blk(x, attn_mask=causal_mask, key_padding_mask=key_padding_mask)

        x = self.ln_f(x)
        # pooled representation: final token's embedding (decoder style)
        pooled = x[:, -1, :]   # (B, d_model)
        return pooled

    def forward(self, input_ids, attention_mask=None):
        pooled = self.encode(input_ids, attention_mask=attention_mask)  # (B, D)
        logits = self.classifier(pooled)  # (B, C)
        return logits
