import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaConfig, RobertaForSequenceClassification
from .bsna import BSNAClassifierBase, bsna_scope, default_ratios, instrument_modules


class BSNARoBERTaBase(BSNAClassifierBase):
    def __init__(self, num_classes, num_subsets, z_dim=256, pretrained=False, model_name="roberta-base"):
        super().__init__(num_subsets, z_dim)
        if pretrained:
            self.backbone = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
        else:
            config = RobertaConfig(num_labels=num_classes)
            self.backbone = RobertaForSequenceClassification(config)
        self.semantic_proj = nn.Linear(self.backbone.config.hidden_size, z_dim)
        def selector(name, module):
            if not isinstance(module, nn.Linear):
                return False
            n = name.lower()
            return "intermediate.dense" in n or "output.dense" in n
        instrument_modules(self.backbone, z_dim, num_subsets, selector, default_ratios("nlp"))

    def encode_input(self, input_ids, attention_mask=None):
        e = self.backbone.roberta.embeddings.word_embeddings(input_ids)
        if attention_mask is None:
            pooled = e.mean(dim=1)
        else:
            m = attention_mask.unsqueeze(-1).to(e.dtype)
            pooled = (e * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        return F.normalize(self.semantic_proj(pooled), dim=-1)

    def forward(self, input_ids, attention_mask=None, return_aux=False):
        z = self.encode_input(input_ids, attention_mask)
        subset, centroid = self.routing(z)
        with bsna_scope(self.stage, z, centroid, subset):
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        logits = out.logits
        if return_aux:
            return logits, self.aux_state(z, subset)
        return logits
