"""
Experiment 8: 神经退行性 Drop 验证实验

要求对齐点：
- target_damage = 0.913（存活率 0.087）
- 从训练进度 start_fraction（默认 0.3）开始，epoch 级逐步累积
- 推理阶段不加任何 drop（纯推理）
- 尽可能与 exp5/exp7 的保存方式对齐，便于单一变量对比
"""

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
from neurodegenerative_drop import NeurodegenerativeDrop
from results import save_metrics, save_model, load_model_if_exists
from train import train_one_epoch, validate, get_warmup_then_multistep
from utils import ensure_dir, get_device, set_random_seeds, setup_model_on_gpus
from visualization import plot_training_curves, plot_layer_activation_rates


def _write_csv(layer_rates: Dict[str, float], path: str, header=("layer", "rate")):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(header))
        for k, v in layer_rates.items():
            w.writerow([k, v])


def _linear_desired_damage(epoch: int, num_epochs: int, start_epoch: int, target_damage: float) -> float:
    if epoch < start_epoch:
        return 0.0
    denom = max(1, num_epochs - start_epoch)  # 让最后一个 epoch progress=1
    progress = float(epoch - start_epoch + 1) / float(denom)
    return float(min(target_damage, target_damage * progress))


def run_experiment(
    experiment_name: str = "exp8_neurodegenerative_drop",
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
    cifar_stem: bool = True,
    # --- 神经退行性 Drop 参数 ---
    target_damage: float = 0.913,
    start_fraction: float = 0.3,
    interval: int = 1,
    path_scopes: Optional[List[str]] = None,   # 默认 mid+late
    block_scopes: Optional[List[str]] = None,  # 默认 late
    path_component: float = 1.0,               # 0=禁用 path；>0 启用
    block_component: float = 1.0,              # 0=禁用 block；>0 启用（仅 late 生效）
    block_group_size: int = 4,
    # --- 激活（权重法） ---
    activation_threshold: float = 0.01,
    activation_threshold_type: str = "absolute",
):
    if gpu_ids is None:
        gpu_ids = []
    if path_scopes is None:
        path_scopes = ["mid", "late"]
    if block_scopes is None:
        block_scopes = ["late"]

    set_random_seeds(seed)
    device = get_device(gpu_ids)
    root = os.path.join("results", experiment_name)
    ensure_dir(root)

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
    baseline_dir = os.path.join(root, "baseline")
    sick_dir = os.path.join(root, "sick_neurodegenerative")
    ensure_dir(baseline_dir)
    ensure_dir(sick_dir)

    # -------- baseline：尽量复用其它实验的 baseline（不属于“整合”，只是复用 checkpoint）--------
    baseline_candidates = [
        os.path.join("results", "exp5_normal_vs_sick", "baseline"),
        os.path.join("results", "exp7_irreversible_damage", "baseline"),
        os.path.join("results", "exp3_brain_simulation", "baseline"),
        baseline_dir,
    ]

    base_model = build_model(model_name=model_name, num_classes=100, pretrained=False, cifar_stem=cifar_stem)
    base_model = setup_model_on_gpus(base_model, gpu_ids=gpu_ids)

    loaded = False
    used_baseline_dir = None
    for cand in baseline_candidates:
        if load_model_if_exists(base_model, output_dir=cand, filename="best_model.pth", device=device):
            loaded = True
            used_baseline_dir = cand
            break

    if not loaded:
        # 训练 baseline（完整 epoch；便于与病脑对齐）
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = SGD(base_model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        scheduler = get_warmup_then_multistep(
            optimizer,
            num_epochs=num_epochs,
            warmup_epochs=5,
            warmup_start_factor=0.1,
            gamma=0.1,
        )

        hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}
        for epoch in range(num_epochs):
            tr_l, tr_a = train_one_epoch(base_model, train_loader, criterion, optimizer, device)
            va_l, va_a = validate(base_model, val_loader, criterion, device)
            cur_lr = optimizer.param_groups[0]["lr"]
            hist["train_loss"].append(tr_l)
            hist["val_loss"].append(va_l)
            hist["train_acc"].append(tr_a)
            hist["val_acc"].append(va_a)
            hist["lr"].append(cur_lr)
            scheduler.step()
            print(f"[Exp8][baseline] Epoch {epoch+1}/{num_epochs} loss={tr_l:.4f}/{va_l:.4f} acc={tr_a:.4f}/{va_a:.4f} lr={cur_lr:.6f}")

        plot_training_curves(hist, output_dir=baseline_dir, prefix="exp8_baseline")
        save_model(base_model, output_dir=baseline_dir, filename="best_model.pth")
        used_baseline_dir = baseline_dir

    # baseline 评估（保存到 exp8/baseline）
    base_cls = evaluate_classification(base_model, test_loader, device, criterion=nn.CrossEntropyLoss())
    (bg_all, bl_all, bg_conv, bl_conv) = compute_weight_based_activation(
        model=base_model,
        threshold=activation_threshold,
        threshold_type=activation_threshold_type,
    )
    base_metrics = {
        "classification": base_cls,
        "activation_weight_based": {
            "all_layers": {"global_activation_rate": bg_all, "layer_activation_rates": bl_all},
            "conv_only": {"global_activation_rate": bg_conv, "layer_activation_rates": bl_conv},
        },
        "config": {"baseline_dir_used": used_baseline_dir},
    }
    save_metrics(base_metrics, output_dir=baseline_dir, filename="summary_metrics.json")
    plot_layer_activation_rates(bl_all, baseline_dir, filename="layer_activation_weight_all.png",
                                title="Baseline Activation (weight, all)")
    plot_layer_activation_rates(bl_conv, baseline_dir, filename="layer_activation_weight_conv.png",
                                title="Baseline Activation (weight, conv-only)")

    # -------- sick（神经退行性 Drop）：若 best_model 已存在则跳过训练 --------
    sick_model = build_model(model_name=model_name, num_classes=100, pretrained=False, cifar_stem=cifar_stem)
    sick_model = setup_model_on_gpus(sick_model, gpu_ids=gpu_ids)

    if load_model_if_exists(sick_model, output_dir=sick_dir, filename="best_model.pth", device=device):
        print(f"[Exp8][sick] Found existing best_model in {sick_dir}, skip training.")
        # 尝试加载状态
        state_path = os.path.join(sick_dir, "neurodrop_state.pth")
        neuro_state = torch.load(state_path, map_location="cpu") if os.path.exists(state_path) else None
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = SGD(sick_model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        scheduler = get_warmup_then_multistep(
            optimizer,
            num_epochs=num_epochs,
            warmup_epochs=5,
            warmup_start_factor=0.1,
            gamma=0.1,
        )

        # 初始化神经退行性 Drop（不可逆）
        neuro = NeurodegenerativeDrop(
            model=sick_model,
            model_name=model_name,
            path_scopes=path_scopes,
            block_scopes=block_scopes,
            block_group_size=block_group_size,
            seed=seed,
        )

        start_epoch = int(round(num_epochs * float(start_fraction)))
        start_epoch = max(0, min(num_epochs - 1, start_epoch))
        interval = max(1, int(interval))

        hist = {
            "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": [],
            "structural_damage": [],
        }

        for epoch in range(num_epochs):
            tr_l, tr_a = train_one_epoch(sick_model, train_loader, criterion, optimizer, device)
            va_l, va_a = validate(sick_model, val_loader, criterion, device)
            cur_lr = optimizer.param_groups[0]["lr"]

            # epoch 末尾施加损伤（影响下一 epoch）
            if epoch >= start_epoch and ((epoch - start_epoch) % interval == 0):
                desired = _linear_desired_damage(epoch, num_epochs, start_epoch, target_damage)
                removed = neuro.apply_uniform_target(
                    target_damage=desired,
                    path_component=path_component,
                    block_component=block_component,
                    block_group_size=block_group_size,
                )
                total_removed = sum(removed.values()) if removed else 0
                print(f"[Exp8][sick] Epoch {epoch+1}: lesion step desired_damage={desired:.4f}, new_removed_channels={total_removed}")

            g_damage, _ = neuro.get_structural_damage()

            hist["train_loss"].append(tr_l)
            hist["val_loss"].append(va_l)
            hist["train_acc"].append(tr_a)
            hist["val_acc"].append(va_a)
            hist["lr"].append(cur_lr)
            hist["structural_damage"].append(g_damage)

            scheduler.step()
            print(f"[Exp8][sick] Epoch {epoch+1}/{num_epochs} loss={tr_l:.4f}/{va_l:.4f} acc={tr_a:.4f}/{va_a:.4f} lr={cur_lr:.6f} damage={g_damage:.4f}")

        # 保存训练曲线/模型/状态
        plot_training_curves(hist, output_dir=sick_dir, prefix="exp8_sick_neurodegenerative")
        save_model(sick_model, output_dir=sick_dir, filename="best_model.pth")

        neuro_state = neuro.state_dict()
        torch.save(neuro_state, os.path.join(sick_dir, "neurodrop_state.pth"))

        # 保存每层 damage（CSV + 图）
        g_damage, layer_damage = neuro.get_structural_damage()
        _write_csv(layer_damage, os.path.join(sick_dir, "layer_structural_damage.csv"), header=("layer", "damage_rate"))
        plot_layer_activation_rates(layer_damage, sick_dir, filename="layer_structural_damage.png",
                                    title="Layer-wise Structural Damage (conv)")

    # sick 评估
    sick_cls = evaluate_classification(sick_model, test_loader, device, criterion=nn.CrossEntropyLoss())
    (sg_all, sl_all, sg_conv, sl_conv) = compute_weight_based_activation(
        model=sick_model, threshold=activation_threshold, threshold_type=activation_threshold_type
    )

    # 若有状态文件，计算一次结构损伤（用于 summary）
    structural = None
    if neuro_state is not None:
        # 需要一个临时 NeuroDrop 来读 mask 并统计（不改变模型）
        tmp = NeurodegenerativeDrop(
            model=sick_model,
            model_name=model_name,
            path_scopes=path_scopes,
            block_scopes=block_scopes,
            block_group_size=block_group_size,
            seed=seed,
        )
        tmp.load_state_dict(neuro_state, strict=False)
        gd, ld = tmp.get_structural_damage()
        structural = {
            "all_layers": {"global_damage_rate": gd, "layer_damage_rates": ld},
            "conv_only": {"global_damage_rate": gd, "layer_damage_rates": ld},
        }

    sick_metrics = {
        "classification": sick_cls,
        "activation_weight_based": {
            "all_layers": {"global_activation_rate": sg_all, "layer_activation_rates": sl_all},
            "conv_only": {"global_activation_rate": sg_conv, "layer_activation_rates": sl_conv},
        },
        "structural_damage": structural,
        "config": {
            "target_damage": target_damage,
            "start_fraction": start_fraction,
            "interval": interval,
            "path_scopes": path_scopes,
            "block_scopes": block_scopes,
            "path_component": path_component,
            "block_component": block_component,
            "block_group_size": block_group_size,
        },
    }
    save_metrics(sick_metrics, output_dir=sick_dir, filename="summary_metrics.json")
    plot_layer_activation_rates(sl_all, sick_dir, filename="layer_activation_weight_all.png",
                                title="Sick Activation (weight, all)")
    plot_layer_activation_rates(sl_conv, sick_dir, filename="layer_activation_weight_conv.png",
                                title="Sick Activation (weight, conv-only)")

    # 总览对比
    base_all = bg_all
    base_conv = bg_conv
    sick_all = sg_all
    sick_conv = sg_conv
    if structural is not None:
        d = structural["all_layers"]["global_damage_rate"]
        expected_all = base_all * (1.0 - d)
        expected_conv = base_conv * (1.0 - d)
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
        "sick_neurodegenerative": {
            "top1_acc": sick_cls.get("top1_acc", None),
            "top5_error": sick_cls.get("top5_error", None),
            "activation_all_global": sick_all,
            "activation_conv_global": sick_conv,
            "structural_damage_all_global": structural["all_layers"]["global_damage_rate"] if structural else None,
        },
        "expected_activation_if_structure_only": {
            "all_layers_global": expected_all,
            "conv_only_global": expected_conv,
        },
    }
    save_metrics(overview, output_dir=root, filename="comparison_overview.json")
    print(f"[Exp8] Done. Results under: {root}")


if __name__ == "__main__":
    import argparse
    from utils import parse_gpu_ids

    p = argparse.ArgumentParser("Exp8: Neurodegenerative Drop Validation")

    p.add_argument("--model", type=str, default="resnet18")
    p.add_argument("--dataset", type=str, default="cifar100")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--gpu-ids", type=str, default="0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--cifar-stem", type=int, default=1)

    p.add_argument("--target-damage", type=float, default=0.913)
    p.add_argument("--start-fraction", type=float, default=0.3)
    p.add_argument("--interval", type=int, default=1)
    p.add_argument("--path-scopes", type=str, default="mid,late")
    p.add_argument("--block-scopes", type=str, default="late")
    p.add_argument("--path-component", type=float, default=1.0)
    p.add_argument("--block-component", type=float, default=1.0)
    p.add_argument("--block-group-size", type=int, default=4)

    p.add_argument("--activation-threshold", type=float, default=0.01)
    p.add_argument("--activation-threshold-type", type=str, default="absolute")

    args = p.parse_args()
    gpu_ids = parse_gpu_ids(args.gpu_ids)

    def _parse_list(s: str) -> List[str]:
        return [x.strip() for x in (s or "").replace("，", ",").split(",") if x.strip()]

    run_experiment(
        experiment_name="exp8_neurodegenerative_drop",
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
        cifar_stem=bool(args.cifar_stem),
        target_damage=args.target_damage,
        start_fraction=args.start_fraction,
        interval=args.interval,
        path_scopes=_parse_list(args.path_scopes),
        block_scopes=_parse_list(args.block_scopes),
        path_component=args.path_component,
        block_component=args.block_component,
        block_group_size=args.block_group_size,
        activation_threshold=args.activation_threshold,
        activation_threshold_type=args.activation_threshold_type,
    )