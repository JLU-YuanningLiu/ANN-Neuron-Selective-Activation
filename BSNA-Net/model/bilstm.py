import torch
import torch.nn as nn
import torch.nn.functional as F
from .bsna import BSNAClassifierBase, BSNAFeatureGate, bsna_scope, default_ratios


class BSNABiLSTM(BSNAClassifierBase):
    def __init__(self, vocab_size, num_classes, num_subsets, embedding_dim=300, hidden_size=256, num_layers=2, dropout=0.5, z_dim=256, padding_idx=0):
        super().__init__(num_subsets, z_dim)
        ratios = default_ratios("nlp")
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.semantic_proj = nn.Linear(embedding_dim, z_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.hidden_gate = BSNAFeatureGate(hidden_size * 2, z_dim, num_subsets, ratios["deep"], axis="last")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def encode_input(self, input_ids, attention_mask=None):
        e = self.embedding(input_ids)
        if attention_mask is None:
            pooled = e.mean(dim=1)
        else:
            m = attention_mask.unsqueeze(-1).to(e.dtype)
            pooled = (e * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        return F.normalize(self.semantic_proj(pooled), dim=-1)

    def forward(self, input_ids, attention_mask=None, return_aux=False):
        e = self.embedding(input_ids)
        if attention_mask is None:
            pooled = e.mean(dim=1)
        else:
            m = attention_mask.unsqueeze(-1).to(e.dtype)
            pooled = (e * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        z = F.normalize(self.semantic_proj(pooled), dim=-1)
        subset, centroid = self.routing(z)
        with bsna_scope(self.stage, z, centroid, subset):
            _, (h, _) = self.lstm(e)
            h = torch.cat([h[-2], h[-1]], dim=-1)
            h = self.hidden_gate(h)
            logits = self.classifier(self.dropout(h))
        if return_aux:
            return logits, self.aux_state(z, subset)
        return logits
