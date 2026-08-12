from importlib import import_module


MODEL_REGISTRY = {
    "resnet50": ("model.resnet50", "BSNAResNet50"),
    "densenet121": ("model.densenet121", "BSNADenseNet121"),
    "efficientnet_b0": ("model.efficientnet_b0", "BSNAEfficientNetB0"),
    "convnext_t": ("model.convnext_t", "BSNAConvNeXtT"),
    "vit_b16": ("model.vit_b16", "BSNAViTB16"),
    "swin_t": ("model.swin_t", "BSNASwinT"),
    "fasttext": ("model.fasttext", "BSNAFastText"),
    "textcnn": ("model.textcnn", "BSNATextCNN"),
    "bilstm": ("model.bilstm", "BSNABiLSTM"),
    "bert_base": ("model.bert_base", "BSNABertBase"),
    "roberta_base": ("model.roberta_base", "BSNARoBERTaBase"),
    "deberta_base": ("model.deberta_base", "BSNADeBERTaBase")
}


def build_model(name, **kwargs):
    module_name, class_name = MODEL_REGISTRY[name]
    cls = getattr(import_module(module_name), class_name)
    return cls(**kwargs)
