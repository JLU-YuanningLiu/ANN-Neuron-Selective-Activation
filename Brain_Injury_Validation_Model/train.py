import copy
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class EarlyStopping:
    """
    Simple early stopping on a monitored metric (e.g. validation loss).
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, min_epochs: int = 0):
        """
        Args:
            patience: How many epochs to wait without improvement before stopping.
            min_delta: Minimum change to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.best: Optional[float] = None
        self.num_bad_epochs: int = 0
        self.should_stop: bool = False
        self._seen_epochs: int = 0

    def step(self, current: float):
        """
        Update early stopping status with a new metric value.
        """
        self._seen_epochs += 1
        if self.best is None or current < self.best - self.min_delta:
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        # >>> 只有在 seen_epochs >= min_epochs 时才允许 early stop
        if self._seen_epochs >= self.min_epochs and self.num_bad_epochs >= self.patience:
            self.should_stop = True


def get_linear_scheduler(optimizer: Optimizer, num_epochs: int):
    """
    Create a linear learning rate scheduler that decays LR from 1.0 to 0.0 over num_epochs.

    Note:
        The absolute LR is controlled by the optimizer; this scheduler only scales it.
    """
    def lr_lambda(epoch: int):
        # epoch ranges from 0 to num_epochs - 1
        return max(0.0, 1.0 - float(epoch) / float(max(1, num_epochs - 1)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def get_cosine_scheduler_with_warmup(optimizer: Optimizer, num_epochs: int, warmup_epochs: int = 5):
    import math
    warmup_epochs = max(0, int(warmup_epochs))
    total = max(1, num_epochs)

    def lr_lambda(epoch: int):
        # epoch: 0..total-1
        if epoch < warmup_epochs and warmup_epochs > 0:
            return float(epoch + 1) / float(warmup_epochs)  # 0->1 线性升温
        # 余弦退火到 0
        progress = (epoch - warmup_epochs) / float(max(1, total - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _three_step_milestones(num_epochs: int, ratios=(0.3, 0.6, 0.8)):
    ms = sorted(set(int(round(num_epochs * r)) for r in ratios))
    # 过滤边界，防止 0 或 = num_epochs
    return [m for m in ms if 0 < m < num_epochs]


def get_multistep_scheduler(optimizer: Optimizer, num_epochs: int,
                                  gamma: float = 0.1, ratios=(0.3, 0.6, 0.8)):
    """
    纯 MultiStepLR（无暖启），默认在 30%/60%/80% 处 * gamma。
    例如：num_epochs=200 -> milestones=[60, 120, 160]
    """
    from torch.optim.lr_scheduler import MultiStepLR
    milestones = _three_step_milestones(num_epochs, ratios)
    return MultiStepLR(optimizer, milestones=milestones, gamma=gamma)


def get_warmup_then_multistep(optimizer: Optimizer, num_epochs: int,
                                    warmup_epochs: int = 5, warmup_start_factor: float = 0.1,
                                    gamma: float = 0.1, ratios=(0.3, 0.6, 0.8)):
    """
    先线性 Warmup，再三段 MultiStepLR。
    - warmup: 从 lr * warmup_start_factor 线性升到原始 lr
    - multistep: 默认 30%/60%/80% 处 * gamma
    """
    from torch.optim.lr_scheduler import LinearLR, MultiStepLR, SequentialLR
    milestones = _three_step_milestones(num_epochs, ratios)
    warmup = LinearLR(optimizer, start_factor=warmup_start_factor, end_factor=1.0, total_iters=warmup_epochs)
    multistep = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    return SequentialLR(optimizer, schedulers=[warmup, multistep], milestones=[warmup_epochs])


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Train for a single epoch.

    Returns:
        (avg_loss, avg_top1_acc)
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size

        _, preds = outputs.max(dim=1)
        total_correct += (preds == targets).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / max(1, total_samples)
    avg_acc = total_correct / max(1, total_samples)

    return avg_loss, avg_acc


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate on a validation or test set.

    Returns:
        (avg_loss, avg_top1_acc)
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, targets)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size

            _, preds = outputs.max(dim=1)
            total_correct += (preds == targets).sum().item()
            total_samples += batch_size

    avg_loss = total_loss / max(1, total_samples)
    avg_acc = total_correct / max(1, total_samples)

    return avg_loss, avg_acc


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    num_epochs: int = 100,
    scheduler=None,
    early_stopping: Optional[EarlyStopping] = None,
) -> Tuple[nn.Module, Dict[str, list], int]:
    """
    Full training loop with optional linear LR scheduler and early stopping.

    Args:
        model: Model to train.
        train_loader: Training dataloader.
        val_loader: Validation dataloader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device.
        num_epochs: Max epochs.
        scheduler: LR scheduler (e.g. from get_linear_scheduler).
        early_stopping: EarlyStopping instance.

    Returns:
        model: The best model (by validation loss).
        history: Dict with training curves.
        best_epoch: Index of best epoch (0-based).
    """
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "lr": [],
    }

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        # Track best model by validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())

        # Early stopping
        if early_stopping is not None:
            early_stopping.step(val_loss)
            if early_stopping.should_stop:
                print(f"[EarlyStopping] Stop at epoch {epoch + 1}/{num_epochs}")
                break

        # Step scheduler at end of epoch
        if scheduler is not None:
            scheduler.step()

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.6f}"
        )

    # Load best weights
    model.load_state_dict(best_model_wts)
    return model, history, best_epoch
