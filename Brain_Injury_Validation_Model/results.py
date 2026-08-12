import csv
import json
import os
from typing import Dict

import torch

from utils import ensure_dir


def save_metrics(metrics: Dict, output_dir: str, filename: str = "metrics.json"):
    """
    Save a metrics dict as JSON.
    """
    ensure_dir(output_dir)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    print(f"[Results] Saved metrics to {path}")


def save_layer_activation_rates(
    layer_rates: Dict[str, float],
    output_dir: str,
    filename: str = "layer_activation_rates.csv",
):
    """
    Save per-layer activation rates to CSV.
    """
    ensure_dir(output_dir)
    path = os.path.join(output_dir, filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "activation_rate"])
        for layer_name, rate in layer_rates.items():
            writer.writerow([layer_name, rate])
    print(f"[Results] Saved layer activation rates to {path}")


def save_model(model: torch.nn.Module, output_dir: str, filename: str = "best_model.pth"):
    """
    Save model weights (state_dict).
    """
    ensure_dir(output_dir)
    path = os.path.join(output_dir, filename)

    state_dict = model.state_dict()
    # If using DataParallel, save underlying module
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()

    torch.save(state_dict, path)
    print(f"[Results] Saved best model weights to {path}")

def load_model_if_exists(
    model: torch.nn.Module,
    output_dir: str,
    filename: str = "best_model.pth",
    device=None,
) -> bool:
    """
    If a saved model exists in output_dir/filename, load its weights into `model`
    and return True. Otherwise do nothing and return False.

    Args:
        model: Model instance to load weights into.
        output_dir: Directory containing the file.
        filename: File name, default 'best_model.pth'.
        device: map_location for torch.load; if None, use 'cpu'.

    Returns:
        bool: True if file was found and loaded, False otherwise.
    """
    path = os.path.join(output_dir, filename)
    if not os.path.exists(path):
        return False

    map_location = device if device is not None else "cpu"
    state_dict = torch.load(path, map_location=map_location)

    # If the model is wrapped by DataParallel, load into underlying module
    if hasattr(model, "module"):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)

    print(f"[Results] Loaded model weights from {path}")
    return True