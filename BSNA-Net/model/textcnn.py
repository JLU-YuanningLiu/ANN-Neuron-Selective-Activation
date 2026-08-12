import torch
import torch.nn as nn
import torch.nn.functional as F
from .bsna import BSNAClassifierBase, BSNAConv1d, BSNAFeatureGate, bsna_scope, default_ratios


class BSNATextCNN(BSNAClassifierBase):
    def __init__(self, vocab_size, num_classes, num_subsets, embedding_dim=300, channels=128, kernels=(3, 4, 5), dropout=0.5, z_dim=256, padding_idx=0):
        super().__init__(num_subsets, z_dim)
        ratios = default_ratios("nlp")
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.semantic_proj = nn.Linear(embedding_dim, z_dim)
        self.embedding_gate = BSNAFeatureGate(embedding_dim, z_dim, num_subsets, ratios["shallow"], axis="last")
        self.convs = nn.ModuleList([
            BSNAConv1d(nn.Conv1d(embedding_dim, channels, k), z_dim, num_subsets, ratios["middle"])
            for k in kernels
        ])
        hidden_dim = channels * len(kernels)
        self.hidden_gate = BSNAFeatureGate(hidden_dim, z_dim, num_subsets, ratios["deep"], axis="last")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def encode_input(self, input_ids, attention_mask=None):
        e = self.embedding(input_ids)
        if attention_mask is None:
            pooled = e.mean(dim=1)
        else:
            m = attention_mask.unsqueeze(-1).to(e.dtype)
            pooled = (e * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        return F.normalize(self.semantic_proj(pooled), dim=-1)

    def forward(self, input_ids, attention_mask=None, return_aux=False):
        raw = self.embedding(input_ids)
        if attention_mask is None:
            pooled_raw = raw.mean(dim=1)
        else:
            m = attention_mask.unsqueeze(-1).to(raw.dtype)
            pooled_raw = (raw * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        z = F.normalize(self.semantic_proj(pooled_raw), dim=-1)
        subset, centroid = self.routing(z)
        with bsna_scope(self.stage, z, centroid, subset):
            e = self.embedding_gate(raw).transpose(1, 2)
            features = [F.relu(conv(e)).amax(dim=-1) for conv in self.convs]
            h = self.hidden_gate(torch.cat(features, dim=1))
            logits = self.classifier(self.dropout(h))
        if return_aux:
            return logits, self.aux_state(z, subset)
        return logits
