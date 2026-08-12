import torch.nn as nn
import timm
from .bsna import BSNAClassifierBase, VisionSemanticEncoder, bsna_scope, default_ratios, instrument_modules


class BSNASwinT(BSNAClassifierBase):
    def __init__(self, num_classes, num_subsets, z_dim=256, pretrained=False):
        super().__init__(num_subsets, z_dim)
        self.semantic_encoder = VisionSemanticEncoder(z_dim)
        self.backbone = timm.create_model("swin_tiny_patch4_window7_224", pretrained=pretrained, num_classes=num_classes, drop_path_rate=0.1)
        def selector(name, module):
            if not isinstance(module, nn.Linear):
                return False
            n = name.lower()
            return "attn.qkv" in n or "attn.proj" in n or "mlp.fc1" in n or "mlp.fc2" in n
        instrument_modules(self.backbone, z_dim, num_subsets, selector, default_ratios("vision"))

    def encode_input(self, x):
        return self.semantic_encoder(x)

    def forward(self, x, return_aux=False):
        z = self.encode_input(x)
        subset, centroid = self.routing(z)
        with bsna_scope(self.stage, z, centroid, subset):
            logits = self.backbone(x)
        if return_aux:
            return logits, self.aux_state(z, subset)
        return logits
