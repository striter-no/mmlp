from mlc.tokenizer import TokenEngine

import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class BahdanauAttention(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v      = nn.Linear(hidden, 1, bias=False)

    def forward(self, query, keys, values, mask=None):
        # query: (B, H), keys: (B, S, H), values: (B, S, H)
        q = self.q_proj(query).unsqueeze(2)  # (B, H, 1)
        k = self.k_proj(keys)                # (B, S, H)

        # (B, S, H) * (B, H, 1) -> (B, S, 1)
        energy = torch.bmm(k, q).squeeze(-1) # (B, S)

        if mask is not None:
            energy = energy.masked_fill(~mask, -1e9)

        attn = F.softmax(energy, dim=-1)             # (B, S)

        # (B, 1, S) * (B, S, H) -> (B, 1, H) -> (B, H)
        context = torch.bmm(attn.unsqueeze(1), values).squeeze(1)
        return context, attn


class AutoregressionNetwork(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_len: int,
        embed_dim: int = 32,
        hidden: int = 512,
        pad_id: int = 0,
        eos_id: int = 0,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_attention: bool = True,
    ):
        super().__init__()
        self.vocab_size   = vocab_size
        self.context_len  = context_len
        self.bos_id       = vocab_size
        self.pad_id       = pad_id
        self.eos_id       = eos_id
        self.use_attention = use_attention

        self.dropout = nn.Dropout(dropout)

        self.embed = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=pad_id)

        self.encoder = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        if use_attention:
            self.attention = BahdanauAttention(hidden)
            head_in = hidden * 2     # [decoder hidden ; context]
        else:
            head_in = hidden

        self.cell = nn.GRUCell(embed_dim, hidden)
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, vocab_size)
        )

    def _make_mask(self, ids: torch.Tensor) -> torch.Tensor:
        return ids != self.pad_id

    def get_n_params(self, only_trainable: bool = True) -> int:
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        q_ids: torch.Tensor,
        target_ids: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.8,
        temperature: float = 1.0,
        top_k: int = 0
    ) -> torch.Tensor:
        B = q_ids.size(0)

        # === ENCODER ===
        q_emb = self.dropout(self.embed(q_ids))
        enc_out, h_enc = self.encoder(q_emb)
        h = h_enc[-1]
        mask = self._make_mask(q_ids) if self.use_attention else None

        # === DECODER ===
        prev_id = torch.full((B,), self.bos_id, dtype=torch.long, device=q_ids.device)

        outputs = []
        generated_ids = []

        finished = torch.zeros(B, dtype=torch.bool, device=q_ids.device)

        use_tf = self.training and target_ids is not None

        for t in range(self.context_len):
            prev_emb = self.dropout(self.embed(prev_id))
            h = self.cell(prev_emb, h)

            if self.use_attention:
                context, _ = self.attention(h, enc_out, enc_out, mask)
                logits = self.head(torch.cat([h, context], dim=-1))
            else:
                logits = self.head(h)

            outputs.append(logits)

            if use_tf and random.random() < teacher_forcing_ratio:
                prev_id = target_ids[:, t]
            else:
                if not use_tf and t > 0:
                    logits[torch.arange(B, device=logits.device), prev_id] -= 10.0

                if top_k > 0:
                    top_logits, top_indices = torch.topk(logits, top_k, dim=-1)
                    probs = torch.softmax(top_logits / temperature, dim=-1)
                    idx = torch.multinomial(probs, 1).squeeze(-1)
                    prev_id = top_indices.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
                else:
                    prev_id = logits.argmax(-1)

                prev_id = torch.where(finished, torch.full_like(prev_id, self.pad_id), prev_id)
                finished = finished | (prev_id == self.eos_id)

                generated_ids.append(prev_id)

        if self.training:
            return torch.stack(outputs, dim=1)       # (B, T, V)
        else:
            return torch.stack(generated_ids, dim=1) # (B, T)

    def load(self, path: str, device: torch.device):
        self.load_state_dict(torch.load(path, map_location=device))

    def save(self, path: str):
        torch.save(self.state_dict(), path)


def text_to_ids(engine: TokenEngine, text: str, max_tokens: int, add_eos: bool = False) -> list[int]:
    ids = engine.tokenize(text)
    if add_eos and len(ids) < max_tokens:
        ids.append(engine.eos_id)

    if len(ids) > max_tokens:
        ids = ids[:max_tokens]
    elif len(ids) < max_tokens:
        ids = ids + [engine.pad_id] * (max_tokens - len(ids))
    return ids


def logits_to_ids(logits: torch.Tensor) -> list[int]:
    if logits.dim() == 3:
        logits = logits.squeeze(0)
    return logits.argmax(dim=-1).tolist()
