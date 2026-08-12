from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def imagenet_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])


def bdd_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])


def build_vision_loaders(root, dataset, batch_size, workers=8, val_ratio=None, seed=0):
    root = Path(root)
    train_tf = imagenet_transforms(True) if dataset == "imagenet" else bdd_transforms(True)
    val_tf = imagenet_transforms(False) if dataset == "imagenet" else bdd_transforms(False)
    train_root = root / "train"
    val_root = root / "val"
    if val_root.exists() and val_ratio is None:
        train_set = ImageFolder(train_root, transform=train_tf)
        val_set = ImageFolder(val_root, transform=val_tf)
    else:
        base = ImageFolder(train_root)
        ratio = 0.2 if val_ratio is None else val_ratio
        n_val = int(round(len(base) * ratio))
        n_train = len(base) - n_val
        generator = torch.Generator().manual_seed(seed)
        train_idx, val_idx = random_split(range(len(base)), [n_train, n_val], generator=generator)
        train_set = ImageFolder(train_root, transform=train_tf)
        val_set = ImageFolder(train_root, transform=val_tf)
        train_set = torch.utils.data.Subset(train_set, train_idx.indices)
        val_set = torch.utils.data.Subset(val_set, val_idx.indices)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, val_loader
