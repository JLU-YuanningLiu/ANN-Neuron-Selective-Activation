"""
Experiment 7: 不可逆结构损伤的病脑模拟（Irreversible Damage Simulation）

动机：
    - 在 Experiment 5 中，训练阶段加入 DropPath/DropBlock 的“病脑模型”反而提升了精度，
      这符合 Drop 系列作为正则化的特性，但不符合 AD 患者“神经元损伤 -> 功能下降”的直觉。
    - AD 更像是：在一段时间内神经结构逐渐、不可逆地损伤（通道永远失效），
      之后推理阶段也保持同样的“缺损结构”。

本实验设计：
    - 基于 Experiment 5 的设定（CIFAR-100 + ResNet18 CIFAR stem + 相同训练策略），
      构建一个“不可逆病脑模型”：
        * 在训练阶段，对指定 scope 的 Conv 层通道进行“不可逆剪枝”：
            - 每到一定 epoch，从仍然存活的通道中随机选择一部分“死亡”；
            - 将这些通道对应的权重/bias 置 0；
            - 在权重和 bias 的梯度上注册 hook，使得这些通道在后续训练中永远不再更新。
        * 训练结束后，该模型的结构性损伤固定，推理阶段不再有随机 Drop。
    - 指标：
        * 分类性能：top-1 准确率、top-5 错误率、loss。
        * 激活指标：基于权重，阈值 = 绝对 0.01，
          统计 Conv+Linear & Conv-only 的全局/分层激活率。
        * 结构损伤指标：统计被剪掉通道的比例（全局 + 分层，仅 Conv）。
        * 对比：
            - baseline 的激活率 A_base
            - 病脑模型的结构损伤率 d_struct
            - 理论“只考虑结构减少”的激活率 A_expected = A_base * (1 - d_struct)
            - 实际激活率 A_sick
          观察 A_sick 是否明显低于 A_expected，以模拟“损伤结构还影响剩余结构活性”的 AD 特征。
"""

import copy
import json
import os
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.optim import SGD

from activations import compute_weight_based_activation
from datasets import build_dataloaders
from eval import evaluate_classification
from models import build_model
from results import (
    save_layer_activation_rates,
    save_metrics,
    save_model,
    load_model_if_exists,
)
from stochastic import resolve_scope_selected_names
from train import (
    train_one_epoch,
    validate,
    get_warmup_then_multistep,
)
from utils import (
    ensure_dir,
    get_device,
    set_random_seeds,
    setup_model_on_gpus,
)
from visualization import (
    plot_training_curves,
    plot_layer_activation_rates,
)


# ----------------- 基础：baseline 训练/复用 -----------------


def _ensure_baseline(
    model_name: str,
    train_loader,
    val_loader,
    device: torch.device,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    out_dir: str,
    gpu_ids: List[int],
):
    """
    确保 baseline best_model 可用；若不存在则训练一次。
    训练策略尽量与实验 5 保持一致。
    """
    ensure_dir(out_dir)
    model = build_model(model_name=model_name, num_classes=100, pretrained=False, cifar_stem=True)
    model = setup_model_on_gpus(model, gpu_ids=gpu_ids)

    if load_model_if_exists(model, output_dir=out_dir, filename="best_model.pth", device=device):
        print(f"[Exp7][baseline] Found existing best_model in {out_dir}, skip training.")
        return

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    # Cosine 方案保留，但与前面实验统一，默认使用 warmup + MultiStepLR
    scheduler = get_warmup_then_multistep(
        optimizer,
        num_epochs=num_epochs,
        warmup_epochs=5,
        warmup_start_factor=0.1,
        gamma=0.1,
    )

    print("[Exp7][baseline] Training...")
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "lr": [],
    }

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        if scheduler is not None:
            scheduler.step()

        print(
            f"[Exp7][baseline] Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.6f}"
        )

    plot_training_curves(history, output_dir=out_dir, prefix="exp7_baseline")
    save_model(model, output_dir=out_dir, filename="best_model.pth")
    print(f"[Exp7][baseline] Training finished. Best model = final epoch (no early stopping).")


# ----------------- 不可逆损伤：状态与操作 -----------------


def _build_lesion_state(
    base_model: nn.Module,
    model_name: str,
    lesion_scopes: List[str],
) -> Dict:
    """
    在指定 scope 的 Conv2d 层上构造“不可逆通道损伤”状态，并注册梯度 hook。

    返回的 lesion_state 包含：
        - "masks": {layer_name -> BoolTensor[out_channels]}  通道是否存活
        - "totals": {layer_name -> int}                     每层总通道数
        - "modules": {layer_name -> module}                 对应的 Conv2d 模块
    """
    selected_names = resolve_scope_selected_names(base_model, model_name, lesion_scopes)
    masks: Dict[str, torch.Tensor] = {}
    totals: Dict[str, int] = {}
    modules: Dict[str, nn.Conv2d] = {}

    for name, module in base_model.named_modules():
        if name not in selected_names:
            continue
        if not isinstance(module, nn.Conv2d):
            # 这里我们只对 Conv2d 做不可逆损伤；FC 层仍然完整保留。
            continue

        out_c = module.out_channels
        mask = torch.ones(out_c, dtype=torch.bool, device=module.weight.device)

        masks[name] = mask
        totals[name] = out_c
        modules[name] = module

        # 在权重上注册 grad hook：对 mask=0 的通道，梯度永远为 0（冻结）
        weight = module.weight

        def _weight_grad_hook(grad, mask_ref=mask):
            # grad shape: [out_c, ...]
            view_shape = (mask_ref.shape[0],) + (1,) * (grad.dim() - 1)
            return grad * mask_ref.view(view_shape).to(grad.device)

        weight.register_hook(_weight_grad_hook)

        # 在 bias 上也注册同样的 grad hook（如果存在）
        if module.bias is not None:
            bias = module.bias

            def _bias_grad_hook(grad, mask_ref=mask):
                return grad * mask_ref.to(grad.device)

            bias.register_hook(_bias_grad_hook)

    print(f"[Exp7][lesion] Prepare irreversible lesion on {len(masks)} conv layers: {list(masks.keys())}")
    return {
        "masks": masks,
        "totals": totals,
        "modules": modules,
    }


def _apply_lesion_step(
    lesion_state: Dict,
    lesion_rate: float,
) -> int:
    """
    在当前存活通道中，以给定比例进行一次不可逆剪枝（通道“死亡”），并将权重/bias 置 0。

    Args:
        lesion_state: 由 _build_lesion_state 构造。
        lesion_rate: 0~1，每次损伤 event 中，当前“存活通道”中被剪掉的比例。

    Returns:
        new_removed: 本次新剪掉的通道数量（所有层总和）。
    """
    masks = lesion_state["masks"]
    modules = lesion_state["modules"]

    if lesion_rate <= 0.0:
        return 0

    total_new_removed = 0

    for name, mask in masks.items():
        module = modules[name]
        alive_idx = torch.nonzero(mask, as_tuple=True)[0]
        alive_count = alive_idx.numel()
        if alive_count == 0:
            continue

        # 本层本次准备剪掉的数量
        num_to_kill = int(round(lesion_rate * float(alive_count)))
        if num_to_kill <= 0:
            continue

        # 为了避免整层被剪空，至少保留 1 个通道
        if alive_count - num_to_kill < 1:
            num_to_kill = max(0, alive_count - 1)
            if num_to_kill == 0:
                continue

        perm = torch.randperm(alive_count, device=alive_idx.device)
        kill_idx = alive_idx[perm[:num_to_kill]]
        if kill_idx.numel() == 0:
            continue

        # 更新 mask：通道死亡
        mask[kill_idx] = False
        total_new_removed += kill_idx.numel()

        # 将死亡通道的权重和 bias 置 0（不可逆）
        with torch.no_grad():
            w = module.weight
            w.data[~mask] = 0
            if module.bias is not None:
                module.bias.data[~mask] = 0

    return total_new_removed


def _compute_structural_damage(lesion_state: Dict) -> Tuple[float, Dict[str, float]]:
    """
    根据 lesion_state 计算：
        - 全局结构损伤率（仅统计参与损伤的 Conv 层）
        - 每层结构损伤率
    """
    masks = lesion_state["masks"]
    totals = lesion_state["totals"]

    total_channels = 0
    total_removed = 0
    per_layer_damage: Dict[str, float] = {}

    for name, mask in masks.items():
        total = totals[name]
        alive = mask.sum().item()
        removed = total - alive
        total_channels += total
        total_removed += removed
        ratio = float(removed) / float(total) if total > 0 else 0.0
        per_layer_damage[name] = ratio

    global_damage = float(total_removed) / float(total_channels) if total_channels > 0 else 0.0
    return global_damage, per_layer_damage


# ----------------- 不可逆病脑模型训练 -----------------


def _train_irreversible_sick_model(
    model: nn.Module,
    model_name: str,
    train_loader,
    val_loader,
    device: torch.device,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    lesion_scopes: List[str],
    lesion_start_fraction: float,
    lesion_interval: int,
    lesion_rate_per_event: float,
    out_dir: str,
) -> Tuple[nn.Module, Dict[str, list], Dict]:
    """
    训练“不可逆病脑模型”：
        - 前若干 epoch 正常训练；
        - 从 lesion_start_fraction * num_epochs 之后，每隔 lesion_interval 个 epoch
          对指定 scope 的 Conv 层进行一次不可逆通道剪枝。

    返回：
        model: 训练完成的模型（最终 epoch 的结构）
        history: 训练曲线（含结构损伤曲线）
        damage_info: {
            "all_layers": {
                "global_damage_rate": float,
                "layer_damage_rates": {layer_name: float}
            },
            "conv_only": { ... }  # 此处与 all_layers 相同，因为只对 Conv 做损伤
        }
    """
    ensure_dir(out_dir)
    base_model = model.module if hasattr(model, "module") else model

    # 构造不可逆损伤状态并注册 grad hook
    lesion_state = _build_lesion_state(base_model, model_name, lesion_scopes)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = get_warmup_then_multistep(
        optimizer,
        num_epochs=num_epochs,
        warmup_epochs=5,
        warmup_start_factor=0.1,
        gamma=0.1,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "lr": [],
        "structural_damage": [],  # 每个 epoch 的全局结构损伤率
    }

    lesion_start_epoch = int(num_epochs * lesion_start_fraction)
    lesion_start_epoch = max(0, min(num_epochs - 1, lesion_start_epoch))
    lesion_interval = max(1, lesion_interval)

    print(f"[Exp7][sick] Irreversible lesion config:")
    print(f"          scopes={lesion_scopes}, start_epoch={lesion_start_epoch}, "
          f"interval={lesion_interval}, rate_per_event={lesion_rate_per_event}")

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]

        # 在 epoch 末尾进行一次（可选）损伤事件：作用于下一 epoch 的训练
        if epoch >= lesion_start_epoch and ((epoch - lesion_start_epoch) % lesion_interval == 0):
            new_removed = _apply_lesion_step(lesion_state, lesion_rate_per_event)
            print(f"[Exp7][sick] Epoch {epoch+1}: irreversible lesion event, "
                  f"new_removed_channels={new_removed}")

        global_damage, _ = _compute_structural_damage(lesion_state)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)
        history["structural_damage"].append(global_damage)

        if scheduler is not None:
            scheduler.step()

        print(
            f"[Exp7][sick] Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, "
            f"LR: {current_lr:.6f}, Global Damage: {global_damage:.4f}"
        )

    # 训练完成后，再精确计算一次结构损伤
    global_damage, layer_damage = _compute_structural_damage(lesion_state)
    damage_info = {
        "all_layers": {
            "global_damage_rate": global_damage,
            "layer_damage_rates": layer_damage,
        },
        "conv_only": {
            "global_damage_rate": global_damage,
            "layer_damage_rates": layer_damage,
        },
    }

    plot_training_curves(history, output_dir=out_dir, prefix="exp7_sick_irreversible")
    save_model(model, output_dir=out_dir, filename="best_model.pth")
    print(f"[Exp7][sick] Training finished. Final global structural damage = {global_damage:.4f}")

    return model, history, damage_info


# ----------------- 评估与结果保存（与实验 5 风格对齐） -----------------


def _eval_and_dump_all(
    tag: str,
    model: nn.Module,
    test_loader,
    device: torch.device,
    out_dir: str,
    activation_threshold: float,
    activation_threshold_type: str,
    structural_damage: Optional[Dict] = None,
) -> Dict:
    """
    评估分类性能 + 基于权重的激活率（all / conv-only），并保存到 out_dir。

    若 structural_damage 不为 None，则一并保存结构损伤指标。
    """
    ensure_dir(out_dir)

    criterion = nn.CrossEntropyLoss()
    cls_metrics = evaluate_classification(model, test_loader, device, criterion=criterion)

    (g_all, l_all, g_conv, l_conv) = compute_weight_based_activation(
        model=model,
        threshold=activation_threshold,
        threshold_type=activation_threshold_type,
    )

    metrics = {
        "classification": cls_metrics,
        "activation_weight_based": {
            "all_layers": {
                "global_activation_rate": g_all,
                "layer_activation_rates": l_all,
            },
            "conv_only": {
                "global_activation_rate": g_conv,
                "layer_activation_rates": l_conv,
            },
        },
    }

    if structural_damage is not None:
        metrics["structural_damage"] = structural_damage

    save_metrics(metrics, output_dir=out_dir, filename="summary_metrics.json")

    # 分层激活率（CSV + PNG）
    save_layer_activation_rates(
        l_all,
        output_dir=out_dir,
        filename=f"layer_activation_rates_weight_all_{tag}.csv",
    )
    save_layer_activation_rates(
        l_conv,
        output_dir=out_dir,
        filename=f"layer_activation_rates_weight_conv_only_{tag}.csv",
    )

    plot_layer_activation_rates(
        l_all,
        output_dir=out_dir,
        filename=f"layer_activation_rates_weight_all_{tag}.png",
        title=f"Activation (weight, all) - {tag}",
    )
    plot_layer_activation_rates(
        l_conv,
        output_dir=out_dir,
        filename=f"layer_activation_rates_weight_conv_only_{tag}.png",
        title=f"Activation (weight, conv-only) - {tag}",
    )

    return metrics


# ----------------- 实验 7 主流程 -----------------


def run_experiment(
    experiment_name: str = "exp7_irreversible_damage",
    model_name: str = "resnet18",
    dataset_name: str = "cifar100",
    data_root: str = "./data",
    batch_size: int = 128,
    num_epochs: int = 200,
    lr: float = 0.1,
    weight_decay: float = 5e-4,
    seed: int = 42,
    gpu_ids: Optional[List[int]] = None,
    image_size: int = 32,
    # 不可逆损伤相关参数（默认与 Exp5 的病脑 scope 接近）
    lesion_scopes: Optional[List[str]] = None,   # 默认 ["mid", "late"]
    lesion_start_fraction: float = 0.3,          # 从整个训练进度的哪个比例开始损伤
    lesion_interval: int = 1,                    # 每多少个 epoch 触发一次损伤事件
    lesion_rate_per_event: float = 0.02,         # 每次损伤事件中，当前存活通道被剪掉的比例
    # 激活（权重法）
    activation_threshold: float = 0.01,
    activation_threshold_type: str = "absolute",
    cifar_stem: bool = True,
):
    if gpu_ids is None:
        gpu_ids = []
    if lesion_scopes is None:
        lesion_scopes = ["mid", "late"]

    set_random_seeds(seed)
    device = get_device(gpu_ids)
    exp_root = os.path.join("results", experiment_name)
    ensure_dir(exp_root)

    print(f"[Exp7] Experiment directory: {exp_root}")
    print(f"[Exp7] Using device: {device}")
    print(f"[Exp7] Model: {model_name}, Dataset: {dataset_name}")
    print(f"[Exp7] Lesion scopes: {lesion_scopes}, "
          f"start_fraction={lesion_start_fraction}, interval={lesion_interval}, "
          f"rate_per_event={lesion_rate_per_event}")
    print(f"[Exp7] Activation (weight-based): threshold={activation_threshold} ({activation_threshold_type})")

    # 数据
    train_loader, val_loader, test_loader = build_dataloaders(
        dataset_name=dataset_name,
        data_root=data_root,
        batch_size=batch_size,
        num_workers=4,
        val_ratio=0.1,
        image_size=image_size,
    )

    # 目录
    baseline_dir = os.path.join(exp_root, "baseline")
    sick_dir = os.path.join(exp_root, "sick_irreversible")
    ensure_dir(baseline_dir)
    ensure_dir(sick_dir)

    # ---------- 1) baseline：优先复用已有 best_model ----------
    # 尝试从其它实验中复用 baseline（如果你已经跑过 Exp5 或 Exp3/Exp1）
    baseline_dir_candidates = [
        os.path.join("results", "exp5_normal_vs_sick", "baseline"),
        os.path.join("results", "exp3_brain_simulation", "baseline"),
        os.path.join("results", "exp1_activation_metrics"),
        baseline_dir,  # 本实验自身的 baseline 目录
    ]

    base_model = build_model(model_name=model_name, num_classes=100, pretrained=False, cifar_stem=cifar_stem)
    base_model = setup_model_on_gpus(base_model, gpu_ids=gpu_ids)

    loaded_baseline = False
    used_baseline_dir = None
    for cand in baseline_dir_candidates:
        if load_model_if_exists(base_model, output_dir=cand, filename="best_model.pth", device=device):
            print(f"[Exp7] Reuse baseline best_model from {cand}")
            loaded_baseline = True
            used_baseline_dir = cand
            break

    if not loaded_baseline:
        print("[Exp7] No existing baseline best_model found; training new baseline in Exp7...")
        _ensure_baseline(
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=num_epochs,
            lr=lr,
            weight_decay=weight_decay,
            out_dir=baseline_dir,
            gpu_ids=gpu_ids,
        )
        # 重新加载
        load_model_if_exists(base_model, output_dir=baseline_dir, filename="best_model.pth", device=device)
        used_baseline_dir = baseline_dir

    # baseline 指标评估（在 Exp7 目录下单独存一份，便于对比）
    baseline_metrics = _eval_and_dump_all(
        tag="baseline",
        model=base_model,
        test_loader=test_loader,
        device=device,
        out_dir=baseline_dir,
        activation_threshold=activation_threshold,
        activation_threshold_type=activation_threshold_type,
        structural_damage=None,
    )

    # ---------- 2) sick_irreversible：不可逆病脑模型 ----------
    sick_model = build_model(model_name=model_name, num_classes=100, pretrained=False, cifar_stem=cifar_stem)
    sick_model = setup_model_on_gpus(sick_model, gpu_ids=gpu_ids)

    damage_info: Optional[Dict] = None
    if load_model_if_exists(sick_model, output_dir=sick_dir, filename="best_model.pth", device=device):
        print(f"[Exp7] Found existing sick_irreversible best_model in {sick_dir}, skip training.")
        # 如果已经有 summary_metrics.json，则直接读取其中的结构损伤信息
        metrics_path = os.path.join(sick_dir, "summary_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            damage_info = cached.get("structural_damage", None)
    else:
        print("[Exp7] Training sick_irreversible model...")
        sick_model, sick_history, damage_info = _train_irreversible_sick_model(
            model=sick_model,
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=num_epochs,
            lr=lr,
            weight_decay=weight_decay,
            lesion_scopes=lesion_scopes,
            lesion_start_fraction=lesion_start_fraction,
            lesion_interval=lesion_interval,
            lesion_rate_per_event=lesion_rate_per_event,
            out_dir=sick_dir,
        )

    sick_metrics = _eval_and_dump_all(
        tag="sick_irreversible",
        model=sick_model,
        test_loader=test_loader,
        device=device,
        out_dir=sick_dir,
        activation_threshold=activation_threshold,
        activation_threshold_type=activation_threshold_type,
        structural_damage=damage_info,
    )

    # ---------- 3) 总览对比：baseline vs sick_irreversible ----------
    base_cls = baseline_metrics["classification"]
    base_act = baseline_metrics["activation_weight_based"]
    sick_cls = sick_metrics["classification"]
    sick_act = sick_metrics["activation_weight_based"]

    base_all = base_act["all_layers"]["global_activation_rate"]
    base_conv = base_act["conv_only"]["global_activation_rate"]

    if damage_info is None:
        struct_all = None
        struct_conv = None
    else:
        struct_all = damage_info["all_layers"]["global_damage_rate"]
        struct_conv = damage_info["conv_only"]["global_damage_rate"]

    if struct_all is not None:
        expected_all = base_all * (1.0 - struct_all)
        expected_conv = base_conv * (1.0 - struct_conv)
    else:
        expected_all = None
        expected_conv = None

    overview = {
        "baseline": {
            "top1_acc": base_cls.get("top1_acc", None),
            "top5_error": base_cls.get("top5_error", None),
            "activation_all_global": base_all,
            "activation_conv_global": base_conv,
        },
        "sick_irreversible": {
            "top1_acc": sick_cls.get("top1_acc", None),
            "top5_error": sick_cls.get("top5_error", None),
            "activation_all_global": sick_act["all_layers"]["global_activation_rate"],
            "activation_conv_global": sick_act["conv_only"]["global_activation_rate"],
            "structural_damage_all_global": struct_all,
            "structural_damage_conv_global": struct_conv,
        },
        "expected_activation_if_structure_only": {
            "all_layers_global": expected_all,
            "conv_only_global": expected_conv,
        },
        "config": {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "seed": seed,
            "image_size": image_size,
            "lesion_scopes": lesion_scopes,
            "lesion_start_fraction": lesion_start_fraction,
            "lesion_interval": lesion_interval,
            "lesion_rate_per_event": lesion_rate_per_event,
            "activation_threshold": activation_threshold,
            "activation_threshold_type": activation_threshold_type,
            "cifar_stem": cifar_stem,
            "baseline_dir_used": used_baseline_dir,
        },
    }

    save_metrics(overview, output_dir=exp_root, filename="comparison_overview.json")
    print("[Exp7] Experiment 7 finished.")
    print(f"[Exp7] Results saved under: {exp_root}")


if __name__ == "__main__":
    import argparse
    from utils import parse_gpu_ids

    parser = argparse.ArgumentParser(description="Experiment 7: Irreversible Damage Brain Simulation")

    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--gpu-ids", type=str, default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=32)

    # 不可逆损伤相关参数（可通过命令行调节）
    parser.add_argument("--lesion-scopes", type=str, default="mid,late",
                        help="Comma-separated scopes for lesion (e.g. 'mid,late').")
    parser.add_argument("--lesion-start-fraction", type=float, default=0.3,
                        help="Fraction of total epochs after which irreversible lesion starts (0~1).")
    parser.add_argument("--lesion-interval", type=int, default=1,
                        help="Number of epochs between lesion events.")
    parser.add_argument("--lesion-rate-per-event", type=float, default=0.02,
                        help="Fraction of currently alive channels to permanently prune at each lesion event.")

    # 激活（权重法）
    parser.add_argument("--activation-threshold", type=float, default=0.01)
    parser.add_argument("--activation-threshold-type", type=str, default="absolute")
    parser.add_argument("--cifar-stem", type=int, default=1, help="Use CIFAR-style stem for ResNet (1=yes,0=no)")

    args = parser.parse_args()
    gpu_ids = parse_gpu_ids(args.gpu_ids)

    def _parse_scopes(s: str) -> List[str]:
        return [x.strip() for x in (s or "").replace("，", ",").split(",") if x.strip()]

    run_experiment(
        experiment_name="exp7_irreversible_damage",
        model_name=args.model,
        dataset_name=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        gpu_ids=gpu_ids,
        image_size=args.image_size,
        lesion_scopes=_parse_scopes(args.lesion_scopes),
        lesion_start_fraction=args.lesion_start_fraction,
        lesion_interval=args.lesion_interval,
        lesion_rate_per_event=args.lesion_rate_per_event,
        activation_threshold=args.activation_threshold,
        activation_threshold_type=args.activation_threshold_type,
        cifar_stem=bool(args.cifar_stem),
    )
