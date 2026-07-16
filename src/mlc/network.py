import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_len, embed_dim, hidden, num_layers, dropout=0.1, pad_id=0):
        super().__init__()
        self.context_len = context_len
        self.pad_id = pad_id

        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.pos_enc = PositionalEncoding(embed_dim, max_len=context_len)

        nhead = 4 if embed_dim % 4 == 0 else 2

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=hidden,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x_ids):
        pad_mask = (x_ids == self.pad_id)

        x = self.embed(x_ids)
        x = self.pos_enc(x)

        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)

        out = self.transformer(x, mask=mask, src_key_padding_mask=pad_mask, is_causal=True)
        logits = self.head(out)
        return logits

    def load(self, path: str, device: torch.device):
        self.load_state_dict(torch.load(path, map_location=device))

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def get_n_params(self, only_trainable=True):
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

def text_to_ids(engine, text, max_tokens, add_eos=False):
    ids = engine.tokenize(text)
    if add_eos and len(ids) < max_tokens:
        ids.append(engine.eos_id)

    if len(ids) > max_tokens:
        ids = ids[:max_tokens]
    elif len(ids) < max_tokens:
        ids = ids + [engine.pad_id] * (max_tokens - len(ids))
    return ids
