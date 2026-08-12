import os
from typing import Dict

import matplotlib.pyplot as plt

from utils import ensure_dir


def plot_training_curves(history: Dict[str, list], output_dir: str, prefix: str = "train"):
    """
    Plot training & validation loss / accuracy curves.

    history keys expected:
        - "train_loss", "val_loss", "train_acc", "val_acc"
    optional:
        - "lr": list of learning rates per epoch
        - "structural_damage": list of global structural damage rates per epoch
    """
    ensure_dir(output_dir)
    epochs = range(1, len(history.get("train_loss", [])) + 1)

    lrs = history.get("lr", [])
    structural_damage = history.get("structural_damage", [])

    #LR突变点
    lr_change_epochs = []
    if lrs and len(lrs) > 1:
        prev_lr = lrs[0]
        for i in range(1, len(lrs)):
            if lrs[i] != prev_lr:
                lr_change_epochs.append(i + 1)  # epoch index (1-based)
                prev_lr = lrs[i]

    # ================= Loss 图 =================
    fig, ax1 = plt.subplots()

    # Loss（左 y 轴）
    ax1.plot(epochs, history.get("train_loss", []), label="Train Loss")
    if "val_loss" in history:
        ax1.plot(epochs, history.get("val_loss", []), label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training / Validation Loss + Structural Damage")

    # 收集 legend handle/label
    lines, labels = ax1.get_legend_handles_labels()

    # Structural damage（右 y 轴）
    ax2 = None
    if structural_damage and len(structural_damage) == len(history.get("train_loss", [])):
        damage_epochs = range(1, len(structural_damage) + 1)
        ax2 = ax1.twinx()
        ax2.plot(damage_epochs, structural_damage, linestyle="--", label="Structural Damage")
        ax2.set_ylabel("Global Structural Damage Rate")
        # 追加 legend
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines += lines2
        labels += labels2

        # 标注 LR 突变点（竖虚线）
        if lr_change_epochs:
            for e in lr_change_epochs:
                ax1.axvline(e, linestyle=":", alpha=0.5)

            # 为 LR step 加一个 legend 项
            from matplotlib.lines import Line2D
            lr_legend = Line2D([0], [0], linestyle=":", color="gray", label="LR step")
            lines.append(lr_legend)
            labels.append("LR step")

    ax1.legend(lines, labels, loc="best")

    loss_path = os.path.join(output_dir, f"{prefix}_loss.png")
    fig.tight_layout()
    fig.savefig(loss_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Viz] Saved loss curve to {loss_path}")

    # ================= Accuracy 图 =================
    fig, ax1 = plt.subplots()

    # Acc（左 y 轴）
    ax1.plot(epochs, history.get("train_acc", []), label="Train Acc")
    if "val_acc" in history:
        plt.plot(epochs, history.get("val_acc", []), label="Val Acc")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Training / Validation Acc + Structural Damage")

    lines, labels = ax1.get_legend_handles_labels()

    # Structural damage（右 y 轴）
    ax2 = None
    if structural_damage and len(structural_damage) == len(history.get("train_loss", [])):
        damage_epochs = range(1, len(structural_damage) + 1)
        ax2 = ax1.twinx()
        ax2.plot(damage_epochs, structural_damage, linestyle="--", label="Structural Damage")
        ax2.set_ylabel("Global Structural Damage Rate")
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines += lines2
        labels += labels2

    # 标注 LR 突变点
    if lr_change_epochs:
        for e in lr_change_epochs:
            ax1.axvline(e, linestyle=":", alpha=0.5)

        from matplotlib.lines import Line2D
        lr_legend = Line2D([0], [0], linestyle=":", color="gray", label="LR step")
        lines.append(lr_legend)
        labels.append("LR step")

    ax1.legend(lines, labels, loc="best")

    acc_path = os.path.join(output_dir, f"{prefix}_acc.png")
    fig.tight_layout()
    fig.savefig(acc_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Viz] Saved accuracy curve to {acc_path}")


def plot_layer_activation_rates(
    layer_rates: Dict[str, float],
    output_dir: str,
    filename: str = "layer_activation_rates.png",
    title: str = "Layer-wise Activation Rates",
):
    """
    Plot a bar chart of layer-wise activation rates.
    """
    ensure_dir(output_dir)
    layers = list(layer_rates.keys())
    rates = [layer_rates[k] for k in layers]

    plt.figure(figsize=(max(6, len(layers) * 0.6), 4))
    plt.bar(range(len(layers)), rates)
    plt.xticks(range(len(layers)), layers, rotation=90)
    plt.ylabel("Activation Rate")
    plt.title(title)
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[Viz] Saved layer activation bar chart to {path}")
