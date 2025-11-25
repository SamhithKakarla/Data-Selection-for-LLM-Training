# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def causal_mask(seq_len, device):
    # mask with True where positions should be masked (i.e., future tokens)
    # MultiheadAttention accepts attn_mask with True at masked positions (PyTorch >=1.11)
    mask = torch.triu(torch.ones((seq_len, seq_len), device=device), diagonal=1).bool()
    return mask  # (seq_len, seq_len)

class GPTBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # x: (B, T, D)
        x_ln = self.ln1(x)
        attn_out, _ = self.attn(x_ln, x_ln, x_ln, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, max_len=64, d_model=128, n_layers=4, n_heads=4, dropout=0.1, num_classes=3):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.max_len = max_len
        self.d_model = d_model
        self.blocks = nn.ModuleList([GPTBlock(d_model, n_heads, dropout=dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)

        # Classification head instead of LM head
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids, attention_mask=None):
        """
        input_ids: (B, T)
        attention_mask: (B, T) with 1 for tokens, 0 for pad (optional)
        returns logits: (B, num_classes)
        """
        B, T = input_ids.shape
        if T > self.max_len:
            raise ValueError(f"Sequence length {T} > model max_len {self.max_len}")

        tok = self.token_emb(input_ids)  # (B, T, D)
        pos_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = tok + self.pos_emb(pos_ids)

        # key_padding_mask: True positions are masked
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0  # (B, T) bool

        # Use bidirectional attention for classification (no causal mask)
        for blk in self.blocks:
            x = blk(x, attn_mask=None, key_padding_mask=key_padding_mask)

        x = self.ln_f(x)

        # Pool the sequence: use mean pooling over non-padded tokens
        if attention_mask is not None:
            # Mask out padding tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(x.size())  # (B, T, D)
            sum_embeddings = torch.sum(x * mask_expanded, dim=1)  # (B, D)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)  # (B, D)
            pooled = sum_embeddings / sum_mask  # (B, D)
        else:
            pooled = x.mean(dim=1)  # (B, D)

        logits = self.classifier(pooled)  # (B, num_classes)
        return logits