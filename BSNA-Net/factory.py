from pathlib import Path
import yaml
import torch
from model import build_model
from data import build_vision_loaders, build_simple_text_loaders, build_transformer_text_loaders


CV_MODELS = {"resnet50", "densenet121", "efficientnet_b0", "convnext_t", "vit_b16", "swin_t"}
SIMPLE_NLP_MODELS = {"fasttext", "textcnn", "bilstm"}
TRANSFORMER_NLP_MODELS = {"bert_base", "roberta_base", "deberta_base"}
MODEL_NAMES = {
    "bert_base": "bert-base-uncased",
    "roberta_base": "roberta-base",
    "deberta_base": "microsoft/deberta-base"
}


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_optimizer(model, cfg):
    name = cfg["optimizer"].lower()
    lr = cfg["learning_rate"]
    wd = cfg.get("weight_decay", 0.0)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=cfg.get("momentum", 0.9), weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    raise ValueError(name)


def build_experiment(config, model_name, data_path, num_classes=None, pretrained=False, workers=4, seed=0):
    model_cfg = config["models"][model_name]
    num_classes = config.get("num_classes") if num_classes is None else num_classes
    num_subsets = config["num_subsets"]
    z_dim = config.get("z_dim", 256)
    if model_name in CV_MODELS:
        if num_classes is None:
            raise ValueError("num_classes")
        train_loader, val_loader = build_vision_loaders(
            data_path,
            "imagenet" if config["dataset"] == "imagenet" else "bdd100k",
            model_cfg["batch_size"],
            workers=workers,
            val_ratio=config.get("validation_ratio"),
            seed=seed
        )
        model = build_model(model_name, num_classes=num_classes, num_subsets=num_subsets, z_dim=z_dim, pretrained=pretrained)
        return model, train_loader, val_loader, "vision"
    if model_name in SIMPLE_NLP_MODELS:
        train_loader, val_loader, vocab = build_simple_text_loaders(
            data_path,
            model_cfg["batch_size"],
            val_ratio=config.get("validation_ratio", 0.1),
            max_length=config.get("max_length", 256),
            vocab_size=config.get("vocab_size", 50000),
            workers=workers,
            seed=seed
        )
        kwargs = {
            "vocab_size": len(vocab),
            "num_classes": num_classes,
            "num_subsets": num_subsets,
            "z_dim": z_dim
        }
        for key in ["embedding_dim", "channels", "kernels", "dropout", "hidden_size", "num_layers"]:
            if key in model_cfg:
                kwargs[key] = model_cfg[key]
        model = build_model(model_name, **kwargs)
        return model, train_loader, val_loader, "nlp"
    if model_name in TRANSFORMER_NLP_MODELS:
        from transformers import AutoTokenizer
        hf_name = MODEL_NAMES[model_name]
        tokenizer = AutoTokenizer.from_pretrained(hf_name)
        train_loader, val_loader = build_transformer_text_loaders(
            data_path,
            tokenizer,
            batch_size=model_cfg["batch_size"],
            val_ratio=config.get("validation_ratio", 0.1),
            max_length=config.get("max_length", 256),
            workers=workers,
            seed=seed
        )
        model = build_model(model_name, num_classes=num_classes, num_subsets=num_subsets, z_dim=z_dim, pretrained=pretrained, model_name=hf_name)
        return model, train_loader, val_loader, "nlp"
    raise ValueError(model_name)
