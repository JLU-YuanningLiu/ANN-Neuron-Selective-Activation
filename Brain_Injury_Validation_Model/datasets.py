from typing import Tuple

from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms


def get_cifar100_transforms(image_size: int = 32):
    """
    Get train/test transforms for CIFAR-100.

    Args:
        image_size: Resize size.

    Returns:
        (transform_train, transform_test)
    """
    # CIFAR-100 statistics
    normalize = transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408],
        std=[0.2675, 0.2565, 0.2761],
    )

    if image_size == 32:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            #transforms.RandomErasing(p=0.1),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform_train = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        transform_test = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            normalize,
        ])
    return transform_train, transform_test


def get_cifar100_dataloaders(
    data_root: str,
    batch_size: int = 128,
    num_workers: int = 4,
    val_ratio: float = 0.1,
    image_size: int = 32,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build DataLoaders for CIFAR-100 with a train/val split.

    Args:
        data_root: Root directory for torchvision datasets.
        batch_size: Batch size.
        num_workers: Number of workers for data loading.
        val_ratio: Fraction of training set used for validation.
        image_size: Resize size.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    transform_train, transform_test = get_cifar100_transforms(image_size=image_size)

    full_train_dataset = datasets.CIFAR100(
        root=data_root,
        train=True,
        download=True,
        transform=transform_train,
    )

    val_size = int(len(full_train_dataset) * val_ratio)
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # For validation, it's conventional not to apply data augmentation (only resize + normalize).
    val_base_dataset = datasets.CIFAR100(
        root=data_root,
        train=True,
        download=False,
        transform=transform_test,  
    )
    val_dataset = Subset(val_base_dataset, val_dataset.indices)


    test_dataset = datasets.CIFAR100(
        root=data_root,
        train=False,
        download=True,
        transform=transform_test,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def build_dataloaders(
    dataset_name: str,
    data_root: str,
    batch_size: int = 128,
    num_workers: int = 4,
    val_ratio: float = 0.1,
    image_size: int = 32,
):
    """
    General entry to build dataloaders by dataset name.

    Currently supports:
        - 'cifar100'

    Args:
        dataset_name: Name of dataset.
        data_root: Root directory for datasets.
        batch_size: Batch size.
        num_workers: Number of workers for data loading.
        val_ratio: Validation split ratio.
        image_size: Resize size.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    name = dataset_name.lower()
    if name == "cifar100":
        return get_cifar100_dataloaders(
            data_root=data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            val_ratio=val_ratio,
            image_size=image_size,
        )
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}. Currently only 'cifar100' is implemented.")
