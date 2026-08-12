import os
import random
from typing import List, Optional

import numpy as np
import torch


def set_random_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_gpu_ids(gpu_ids_str: Optional[str]) -> List[int]:
    """
    Parse a GPU id string like "0,1" into a list of ints.
    If gpu_ids_str is None, empty or "-1", return empty list (use CPU).
    """
    if gpu_ids_str is None:
        return []
    gpu_ids_str = gpu_ids_str.strip()
    if gpu_ids_str == "" or gpu_ids_str == "-1":
        return []
    ids = []
    for part in gpu_ids_str.split(","):
        part = part.strip()
        if part == "":
            continue
        try:
            idx = int(part)
            if idx >= 0:
                ids.append(idx)
        except ValueError:
            continue
    return ids


def get_device(gpu_ids: List[int]) -> torch.device:
    """
    Get the main torch.device given GPU ids.
    """
    if torch.cuda.is_available() and len(gpu_ids) > 0:
        return torch.device(f"cuda:{gpu_ids[0]}")
    return torch.device("cpu")


def setup_model_on_gpus(model: torch.nn.Module, gpu_ids: List[int]) -> torch.nn.Module:
    """
    Move model to device and wrap with DataParallel if multiple GPUs are used.
    """
    if torch.cuda.is_available() and len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    device = get_device(gpu_ids)
    model = model.to(device)
    return model


def ensure_dir(path: str):
    """
    Create directory if it does not exist.
    """
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
