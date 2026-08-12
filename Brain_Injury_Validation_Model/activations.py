from typing import Dict, List, Tuple, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader


def _get_conv_linear_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """
    Collect all Conv2d and Linear layers with their names.
    """
    layers: List[Tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layers.append((name, module))
    return layers


def _compute_weight_channel_metrics(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    For each conv/linear layer, compute a per-output-channel weight magnitude metric.

    Returns:
        Dict: layer_name -> tensor of shape [out_channels or out_features]
              containing mean absolute weight per output channel.
    """
    metrics: Dict[str, torch.Tensor] = {}
    for name, module in _get_conv_linear_layers(model):
        weight = module.weight.data.abs()
        if isinstance(module, nn.Conv2d):
            # [out_c, in_c, kH, kW] -> [out_c]
            per_channel = weight.view(weight.size(0), -1).mean(dim=1)
        elif isinstance(module, nn.Linear):
            # [out_features, in_features] -> [out_features]
            per_channel = weight.mean(dim=1)
        else:
            continue
        metrics[name] = per_channel.cpu()
    return metrics


def compute_weight_based_activation(
    model: nn.Module,
    threshold: float = 0.01,
    threshold_type: str = "absolute",
) -> Tuple[float, Dict[str, float], float, Dict[str, float]]:
    """
    Weight-based activation ratio.

    定义（在当前实现中的具体化）：
        - 单位：每一层的每一个输出通道（Conv2d 的 out_channel，Linear 的 out_features）。
        - 对每一层的每一个输出通道，计算其权重张量的 |w| 均值，作为该通道的“激活值”。
        - 若该激活值 > 阈值，则认为该通道是“激活”的。
        - 激活率 = 激活通道数量 / 总通道数量（全局 or 分层）。

    Args:
        model: 已训练或正在评估的模型（只统计 Conv2d 和 Linear 层）。
        threshold: 激活阈值（当前实验使用绝对阈值 0.01）。
        threshold_type: "absolute" 或 "relative"。
            - "absolute": 直接比较激活值和 threshold。
            - "relative":实际阈值 = threshold * 全局平均激活值，其中全局平均激活值是所有 Conv/Linear 通道激活值的平均。

    Returns:
        global_all, layer_rates_all, global_conv_only, layer_rates_conv_only
    """
    weight_metrics = _compute_weight_channel_metrics(model)  # name -> [C]
    layers = _get_conv_linear_layers(model)
    conv_layer_names = {name for name, m in layers if isinstance(m, nn.Conv2d)}

    # 收集所有通道激活值，用于 relative 阈值计算
    all_metrics_list = []
    for per_channel in weight_metrics.values():
        if per_channel is None:
            continue
        all_metrics_list.append(per_channel.view(-1))

    if len(all_metrics_list) == 0:
        return 0.0, {}, 0.0, {}

    all_metrics = torch.cat(all_metrics_list)

    if threshold_type == "absolute":
        threshold_value = float(threshold)
    elif threshold_type == "relative":
        if threshold <= 0:
            raise ValueError(f"Relative threshold must be > 0, got {threshold}")
        global_mean = all_metrics.mean().item()
        threshold_value = float(threshold * global_mean)
    else:
        raise ValueError(f"Unsupported threshold_type: {threshold_type}")

    layer_rates_all: Dict[str, float] = {}
    layer_rates_conv: Dict[str, float] = {}

    total_active_all = 0
    total_channels_all = 0
    total_active_conv = 0
    total_channels_conv = 0

    for name, per_channel in weight_metrics.items():
        if per_channel is None or per_channel.numel() == 0:
            continue

        active_channels = (per_channel > threshold_value).sum().item()
        num_channels = per_channel.numel()
        rate = active_channels / max(1, num_channels)
        layer_rates_all[name] = float(rate)

        total_active_all += active_channels
        total_channels_all += num_channels

        if name in conv_layer_names:
            layer_rates_conv[name] = float(rate)
            total_active_conv += active_channels
            total_channels_conv += num_channels

    global_all = float(total_active_all / max(1, total_channels_all))
    global_conv = float(total_active_conv / max(1, total_channels_conv)) if total_channels_conv > 0 else 0.0

    return global_all, layer_rates_all, global_conv, layer_rates_conv


def compute_forward_based_activation(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.01,
    threshold_type: str = "absolute",
    max_batches: Optional[int] = None,
) -> Tuple[float, Dict[str, float], float, Dict[str, float]]:
    """
    Forward-activation-based activation ratio.

    定义（当前实现）：
        - 在推理阶段，对给定数据集做前向传播。
        - 对每一层的每一个输出通道，统计其输出激活的 |a| 的均值（在 N、H、W 维度上取平均）。
        - 若该均值 > 阈值，则认为该通道是“激活”的。
        - 激活率 = 激活通道数量 / 总通道数量。

    Args:
        model: 模型。
        dataloader: 用于计算激活指标的数据集（通常使用验证集或测试集）。
        device: 设备。
        threshold: 激活阈值（绝对阈值）。
        threshold_type: "absolute" 或 "relative"。
        max_batches: 若不为 None，则只在前 max_batches 个 batch 上统计，用于加速试验。

    Returns:
        global_all, layer_rates_all, global_conv_only, layer_rates_conv_only
    """
    layers = _get_conv_linear_layers(model)
    conv_layer_names = {name for name, m in layers if isinstance(m, nn.Conv2d)}

    activation_sums: Dict[str, torch.Tensor] = {}
    activation_counts: Dict[str, torch.Tensor] = {}
    handles = []

    def _make_hook(layer_name: str, module: nn.Module):
        def hook(module, inp, out):
            with torch.no_grad():
                if not isinstance(out, torch.Tensor):
                    return
                out_abs = out.detach().abs()
                if isinstance(module, nn.Conv2d):
                    sum_per_channel = out_abs.sum(dim=(0, 2, 3))
                    num_per_channel = torch.ones_like(sum_per_channel) * (
                            out_abs.size(0) * out_abs.size(2) * out_abs.size(3)
                    )
                elif isinstance(module, nn.Linear):
                    sum_per_channel = out_abs.sum(dim=0)
                    num_per_channel = torch.ones_like(sum_per_channel) * out_abs.size(0)
                else:
                    return

                if layer_name not in activation_sums:
                    activation_sums[layer_name] = sum_per_channel.detach().cpu()
                    activation_counts[layer_name] = num_per_channel.detach().cpu()
                else:
                    activation_sums[layer_name] += sum_per_channel.detach().cpu()
                    activation_counts[layer_name] += num_per_channel.detach().cpu()

        return hook

    for name, module in layers:
        handles.append(module.register_forward_hook(_make_hook(name, module)))

    model.eval()
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            _ = model(images)
            if max_batches is not None and (batch_idx + 1) >= max_batches:
                break

    for h in handles:
        h.remove()

    per_layer_means: Dict[str, torch.Tensor] = {}
    all_means_list = []

    for name, sums in activation_sums.items():
        counts = activation_counts[name]
        mean_per_channel = sums / torch.clamp(counts, min=1.0)
        per_layer_means[name] = mean_per_channel
        all_means_list.append(mean_per_channel.view(-1))

    if len(all_means_list) == 0:
        return 0.0, {}, 0.0, {}

    all_means = torch.cat(all_means_list)

    if threshold_type == "absolute":
        threshold_value = float(threshold)
    elif threshold_type == "relative":
        if threshold <= 0:
            raise ValueError(f"Relative threshold must be > 0, got {threshold}")
        global_mean = all_means.mean().item()
        threshold_value = float(threshold * global_mean)
    else:
        raise ValueError(f"Unsupported threshold_type: {threshold_type}")

    layer_rates_all: Dict[str, float] = {}
    layer_rates_conv: Dict[str, float] = {}
    total_active_all = 0
    total_channels_all = 0
    total_active_conv = 0
    total_channels_conv = 0

    for name, mean_per_channel in per_layer_means.items():
        active_channels = (mean_per_channel > threshold_value).sum().item()
        num_channels = mean_per_channel.numel()
        rate = active_channels / max(1, num_channels)
        layer_rates_all[name] = float(rate)

        total_active_all += active_channels
        total_channels_all += num_channels

        if name in conv_layer_names:
            layer_rates_conv[name] = float(rate)
            total_active_conv += active_channels
            total_channels_conv += num_channels

    global_all = float(total_active_all / max(1, total_channels_all))
    global_conv = float(total_active_conv / max(1, total_channels_conv)) if total_channels_conv > 0 else 0.0

    return global_all, layer_rates_all, global_conv, layer_rates_conv


def compute_combined_weight_forward_activation(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.01,
    threshold_type: str = "absolute",
    max_batches: Optional[int] = None,
) -> Tuple[float, Dict[str, float], float, Dict[str, float]]:
    """
    Combined weight + forward-activation based activation ratio.

    当前实现的具体化方案：
        - weight_metric: 与 weight-based 相同，对每一层每个输出通道，计算 |w| 的均值。
        - forward_metric: 与 forward-based 相同，对每一层每个输出通道，计算 |a| 的均值。
        - combined_metric = weight_metric * forward_metric（逐通道相乘）。
        - 若 combined_metric > 阈值，则认为该通道是“激活”的。
        - 激活率 = 激活通道数量 / 总通道数量。

    注意：
        这里只是对 “权重 + 激活联合判定” 的一种合理实现，后续若你有更精细的定义，
        可以在保持接口不变的前提下替换具体实现逻辑。

    Args:
        model: 模型。
        dataloader: 用于计算激活指标的数据集。
        device: 设备。
        threshold: 激活阈值。
        threshold_type: "absolute" 或 "relative"。
        max_batches: 若不为 None，则只在前 max_batches 个 batch 上统计。

    Returns:
        global_all, layer_rates_all, global_conv_only, layer_rates_conv_only
    """
    # weight metrics
    weight_metrics = _compute_weight_channel_metrics(model)

    layers = _get_conv_linear_layers(model)
    conv_layer_names = {name for name, m in layers if isinstance(m, nn.Conv2d)}

    # forward metrics
    activation_sums: Dict[str, torch.Tensor] = {}
    activation_counts: Dict[str, torch.Tensor] = {}
    handles = []

    def _make_hook(layer_name: str, module: nn.Module):
        def hook(module, inp, out):
            with torch.no_grad():
                if not isinstance(out, torch.Tensor):
                    return
                out_abs = out.detach().abs()
                if isinstance(module, nn.Conv2d):
                    sum_per_channel = out_abs.sum(dim=(0, 2, 3))
                    num_per_channel = torch.ones_like(sum_per_channel) * (
                            out_abs.size(0) * out_abs.size(2) * out_abs.size(3)
                    )
                elif isinstance(module, nn.Linear):
                    sum_per_channel = out_abs.sum(dim=0)
                    num_per_channel = torch.ones_like(sum_per_channel) * out_abs.size(0)
                else:
                    return

                if layer_name not in activation_sums:
                    activation_sums[layer_name] = sum_per_channel.detach().cpu()
                    activation_counts[layer_name] = num_per_channel.detach().cpu()
                else:
                    activation_sums[layer_name] += sum_per_channel.detach().cpu()
                    activation_counts[layer_name] += num_per_channel.detach().cpu()

        return hook

    for name, module in layers:
        handles.append(module.register_forward_hook(_make_hook(name, module)))

    model.eval()
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            _ = model(images)
            if max_batches is not None and (batch_idx + 1) >= max_batches:
                break

    for h in handles:
        h.remove()

    combined_metrics_per_layer: Dict[str, torch.Tensor] = {}
    all_combined_list = []

    for name, sums in activation_sums.items():
        counts = activation_counts[name]
        mean_act = sums / torch.clamp(counts, min=1.0)

        if name not in weight_metrics:
            continue
        w_metric = weight_metrics[name]
        if w_metric.numel() != mean_act.numel():
            continue

        combined_metric = w_metric * mean_act
        combined_metrics_per_layer[name] = combined_metric
        all_combined_list.append(combined_metric.view(-1))

    if len(all_combined_list) == 0:
        return 0.0, {}, 0.0, {}

    all_combined = torch.cat(all_combined_list)

    if threshold_type == "absolute":
        threshold_value = float(threshold)
    elif threshold_type == "relative":
        if threshold <= 0:
            raise ValueError(f"Relative threshold must be > 0, got {threshold}")
        global_mean = all_combined.mean().item()
        threshold_value = float(threshold * global_mean)
    else:
        raise ValueError(f"Unsupported threshold_type: {threshold_type}")

    layer_rates_all: Dict[str, float] = {}
    layer_rates_conv: Dict[str, float] = {}
    total_active_all = 0
    total_channels_all = 0
    total_active_conv = 0
    total_channels_conv = 0

    for name, combined_metric in combined_metrics_per_layer.items():
        active_channels = (combined_metric > threshold_value).sum().item()
        num_channels = combined_metric.numel()
        rate = active_channels / max(1, num_channels)
        layer_rates_all[name] = float(rate)

        total_active_all += active_channels
        total_channels_all += num_channels

        if name in conv_layer_names:
            layer_rates_conv[name] = float(rate)
            total_active_conv += active_channels
            total_channels_conv += num_channels

    global_all = float(total_active_all / max(1, total_channels_all))
    global_conv = float(total_active_conv / max(1, total_channels_conv)) if total_channels_conv > 0 else 0.0

    return global_all, layer_rates_all, global_conv, layer_rates_conv
