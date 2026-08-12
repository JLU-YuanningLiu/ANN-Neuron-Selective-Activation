import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights
from .bsna import BSNAClassifierBase, VisionSemanticEncoder, bsna_scope, default_ratios, instrument_modules


class BSNADenseNet121(BSNAClassifierBase):
    def __init__(self, num_classes, num_subsets, z_dim=256, pretrained=False):
        super().__init__(num_subsets, z_dim)
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        self.semantic_encoder = VisionSemanticEncoder(z_dim)
        self.backbone = densenet121(weights=weights)
        self.backbone.classifier = nn.Linear(self.backbone.classifier.in_features, num_classes)
        selector = lambda name, module: isinstance(module, nn.Conv2d)
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
