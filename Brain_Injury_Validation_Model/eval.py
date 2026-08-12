from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader


def accuracy_topk(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)):
    """
    Compute the number of correct predictions for the specified values of k.
    Returns raw counts, NOT normalized by batch size.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.item())
    return res


def evaluate_classification(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module = None,
) -> Dict[str, float]:
    """
    Evaluate a classification model: top-1 accuracy, top-5 error rate, and optional loss.

    Args:
        model: Trained model.
        dataloader: DataLoader for evaluation.
        device: Device.
        criterion: Optional loss function.

    Returns:
        Dictionary with keys:
            - "top1_acc"
            - "top5_error"
            - "loss" (if criterion is provided)
    """
    model.eval()
    total_top1 = 0.0
    total_top5 = 0.0
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            batch_size = images.size(0)

            top1, top5 = accuracy_topk(outputs, targets, topk=(1, 5))
            total_top1 += top1
            total_top5 += top5
            total_samples += batch_size

            if criterion is not None:
                loss = criterion(outputs, targets)
                total_loss += loss.item() * batch_size

    metrics = {}
    if total_samples > 0:
        top1_acc = total_top1 / total_samples
        top5_acc = total_top5 / total_samples
        top5_error = 1.0 - top5_acc
    else:
        top1_acc = 0.0
        top5_error = 1.0

    metrics["top1_acc"] = top1_acc
    metrics["top5_error"] = top5_error

    if criterion is not None and total_samples > 0:
        metrics["loss"] = total_loss / total_samples

    return metrics
