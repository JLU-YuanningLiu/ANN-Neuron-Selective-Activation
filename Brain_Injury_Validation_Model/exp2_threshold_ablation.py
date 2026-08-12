"""
Experiment 2: 激活阈值消融实验

基于 Experiment 1 的基础上，固定：
    - 模型（例如 ResNet-34）
    - 数据集（CIFAR-100）
    - 训练策略（学习率、早停、线性衰减等）
    - best model 复用逻辑
只改变：
    - 激活方式：weight / forward / combined
    - 阈值类型：absolute / relative
    - 阈值取值：多个 absolute 值 & 多个 relative 值

相对阈值定义：
    - 对某种激活方式，收集所有 Conv/Linear 通道的“激活值” m_i；
    - 记其全局平均为 μ；
    - 给定相对阈值参数 t（例如 0.5, 1.0, 2.0）；
    - 实际使用的数值阈值为 λ = t * μ；
    - 若 m_i > λ，则该通道视为激活。
"""

import os
from typing import List, Optional, Dict

import torch
from torch import nn
from torch.optim import SGD

from activations import (
    compute_combined_weight_forward_activation,
    compute_forward_based_activation,
    compute_weight_based_activation,
)
from datasets import build_dataloaders
from eval import evaluate_classification
from models import build_model
from results import (
    save_layer_activation_rates,
    save_metrics,
    save_model,
    load_model_if_exists,
)
from train import EarlyStopping, get_linear_scheduler, get_cosine_scheduler_with_warmup, get_multistep_scheduler, get_warmup_then_multistep, train_model
from utils import ensure_dir, get_device, set_random_seeds, setup_model_on_gpus


def _format_threshold_for_filename(value: float) -> str:
    """
    将浮点数阈值转换为适合文件名的字符串：
        0.01 -> 0p01
        2.0  -> 2p0
    """
    return str(value).replace(".", "p")


def run_experiment(
    experiment_name: str = "exp2_threshold_ablation",
    model_name: str = "resnet18",
    dataset_name: str = "cifar100",
    data_root: str = "./data",
    batch_size: int = 128,
    num_epochs: int = 200,
    patience: int = 10,
    lr: float = 0.1,
    weight_decay: float = 5e-4,
    seed: int = 42,
    gpu_ids: Optional[List[int]] = None,
    image_size: int = 32,
    absolute_thresholds: Optional[List[float]] = None,
    relative_thresholds: Optional[List[float]] = None,
    max_batches_for_activation: Optional[int] = None,
    cifar_stem: bool = True,
):
    """
    第二个实验：激活阈值消融。

    参数说明中与 Experiment 1 相同的部分不再赘述。
    新增参数：
        absolute_thresholds:
            需要扫描的绝对阈值列表，例如 [0.001, 0.005, 0.01, 0.02, 0.05]
        relative_thresholds:
            需要扫描的相对阈值列表，例如 [0.5, 1.0, 2.0]
            注意这里的相对阈值 t 在内部会被解释为 λ = t * 全局平均激活值。
        max_batches_for_activation:
            为了加速，可以限制前向统计激活时使用的 batch 数；
            默认为 None，表示使用整个 test 数据集。
    """
    if gpu_ids is None:
        gpu_ids = []

    # 默认阈值列表（你可以之后自由修改或通过命令行传入）
    if absolute_thresholds is None:
        absolute_thresholds = [0.001, 0.005, 0.01, 0.02, 0.05]
    if relative_thresholds is None:
        relative_thresholds = [0.5, 1.0, 2.0]

    set_random_seeds(seed)
    device = get_device(gpu_ids)

    exp_dir = os.path.join("results", experiment_name)
    ensure_dir(exp_dir)
    # 实验1的目录
    exp1_dir = os.path.join("results", "exp1_activation_metrics")

    print(f"[Exp2] Experiment directory: {exp_dir}")
    print(f"[Exp2] Using device: {device}")
    print(f"[Exp2] Model: {model_name}, Dataset: {dataset_name}")
    print(f"[Exp2] Absolute thresholds: {absolute_thresholds}")
    print(f"[Exp2] Relative thresholds: {relative_thresholds}")

    # --------------------
    # 1. 数据
    # --------------------
    print("[Exp2] Building dataloaders...")
    train_loader, val_loader, test_loader = build_dataloaders(
        dataset_name=dataset_name,
        data_root=data_root,
        batch_size=batch_size,
        num_workers=4,
        val_ratio=0.1,
        image_size=image_size,
    )

    # --------------------
    # 2. 模型
    # --------------------
    print("[Exp2] Building model...")
    num_classes = 100  # CIFAR-100
    model = build_model(model_name=model_name, num_classes=num_classes, pretrained=False, cifar_stem=cifar_stem)
    model = setup_model_on_gpus(model, gpu_ids=gpu_ids)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # --------------------
    # 3. 若已有 best_model 则跳过训练
    # --------------------
    history = None
    best_epoch = None

    loaded_from_exp1 = load_model_if_exists(
        model,
        output_dir=exp1_dir,
        filename="best_model.pth",
        device=device,
    )

    if loaded_from_exp1:
        print("[Exp2] Found best model from Experiment 1. Skip training and use loaded weights.")
    else:
        print("[Exp2] No best model from Experiment 1. Start training (and save baseline model for both Exp1 & Exp2)...")

        # 确保 exp1 目录存在，用于保存 baseline 模型
        ensure_dir(exp1_dir)

        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        #scheduler = get_linear_scheduler(optimizer, num_epochs=num_epochs)
        #scheduler = get_cosine_scheduler_with_warmup(optimizer, num_epochs=num_epochs, warmup_epochs=5)
        scheduler = get_warmup_then_multistep(optimizer, num_epochs=num_epochs,
                                                    warmup_epochs=5, warmup_start_factor=0.1,
                                                    gamma=0.1)

        early_stopping = EarlyStopping(patience=patience, min_delta=0.0, min_epochs=161)

        model, history, best_epoch = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=num_epochs,
            scheduler=scheduler,
            early_stopping=early_stopping,
        )
        print(f"[Exp2] Training finished. Best epoch (0-based): {best_epoch}")

        from visualization import plot_training_curves

        plot_training_curves(history, output_dir=exp_dir, prefix="exp2")
        # baseline 模型统一保存到 Experiment 1 的目录下
        save_model(model, output_dir=exp1_dir, filename="best_model.pth")

    # --------------------
    # 4. 分类性能评估（在 test 集上，只做一次）
    # --------------------
    print("[Exp2] Evaluating classification performance on test set...")
    classification_metrics = evaluate_classification(
        model=model,
        dataloader=test_loader,
        device=device,
        criterion=criterion,
    )
    print(f"[Exp2] Classification metrics: {classification_metrics}")

    # --------------------
    # 5. 阈值消融：三种激活方式 × 多个阈值
    # --------------------
    from visualization import plot_layer_activation_rates

    activation_modes = ["weight", "forward", "combined"]

    threshold_results: Dict[str, Dict] = {}

    for mode in activation_modes:
        print(f"[Exp2] Processing activation mode: {mode}")
        mode_result = {
            "absolute": {},
            "relative": {},
        }

        # ---------- 5.1 绝对阈值 ----------
        for t_abs in absolute_thresholds:
            print(f"[Exp2]  Activation mode={mode}, absolute threshold={t_abs}")
            if mode == "weight":
                (g_all, layer_all,
                 g_conv, layer_conv) = compute_weight_based_activation(
                    model=model,
                    threshold=t_abs,
                    threshold_type="absolute",
                )
            elif mode == "forward":
                (g_all, layer_all,
                 g_conv, layer_conv) = compute_forward_based_activation(
                    model=model,
                    dataloader=test_loader,
                    device=device,
                    threshold=t_abs,
                    threshold_type="absolute",
                    max_batches=max_batches_for_activation,
                )
            elif mode == "combined":
                (g_all, layer_all,
                 g_conv, layer_conv) = compute_combined_weight_forward_activation(
                    model=model,
                    dataloader=test_loader,
                    device=device,
                    threshold=t_abs,
                    threshold_type="absolute",
                    max_batches=max_batches_for_activation,
                )
            else:
                raise ValueError(f"Unknown activation mode: {mode}")

            key = str(t_abs)
            mode_result["absolute"][key] = {
                "all_layers": {
                    "global_activation_rate": g_all,
                    "layer_activation_rates": layer_all,
                },
                "conv_only": {
                    "global_activation_rate": g_conv,
                    "layer_activation_rates": layer_conv,
                },
            }

            tag = _format_threshold_for_filename(t_abs)

            # all layers
            base_name_all = f"layer_activation_rates_{mode}_abs_{tag}"
            save_layer_activation_rates(
                layer_all,
                output_dir=exp_dir,
                filename=base_name_all + ".csv",
            )
            plot_layer_activation_rates(
                layer_all,
                output_dir=exp_dir,
                filename=base_name_all + ".png",
                title=f"Layer-wise Activation Rates ({mode}, abs={t_abs}, all layers)",
            )

            # conv only
            base_name_conv = f"layer_activation_rates_{mode}_abs_{tag}_conv_only"
            save_layer_activation_rates(
                layer_conv,
                output_dir=exp_dir,
                filename=base_name_conv + ".csv",
            )
            plot_layer_activation_rates(
                layer_conv,
                output_dir=exp_dir,
                filename=base_name_conv + ".png",
                title=f"Layer-wise Activation Rates ({mode}, abs={t_abs}, conv only)",
            )

        # ---------- 5.2 相对阈值 ----------
        for t_rel in relative_thresholds:
            print(f"[Exp2]  Activation mode={mode}, relative threshold={t_rel}")
            if mode == "weight":
                (g_all, layer_all,
                 g_conv, layer_conv) = compute_weight_based_activation(
                    model=model,
                    threshold=t_rel,
                    threshold_type="relative",
                )
            elif mode == "forward":
                (g_all, layer_all,
                 g_conv, layer_conv) = compute_forward_based_activation(
                    model=model,
                    dataloader=test_loader,
                    device=device,
                    threshold=t_rel,
                    threshold_type="relative",
                    max_batches=max_batches_for_activation,
                )
            elif mode == "combined":
                (g_all, layer_all,
                 g_conv, layer_conv) = compute_combined_weight_forward_activation(
                    model=model,
                    dataloader=test_loader,
                    device=device,
                    threshold=t_rel,
                    threshold_type="relative",
                    max_batches=max_batches_for_activation,
                )
            else:
                raise ValueError(f"Unknown activation mode: {mode}")

            key = str(t_rel)
            mode_result["relative"][key] = {
                "all_layers": {
                    "global_activation_rate": g_all,
                    "layer_activation_rates": layer_all,
                },
                "conv_only": {
                    "global_activation_rate": g_conv,
                    "layer_activation_rates": layer_conv,
                },
            }

            tag = _format_threshold_for_filename(t_rel)

            # all layers
            base_name_all = f"layer_activation_rates_{mode}_rel_{tag}"
            save_layer_activation_rates(
                layer_all,
                output_dir=exp_dir,
                filename=base_name_all + ".csv",
            )
            plot_layer_activation_rates(
                layer_all,
                output_dir=exp_dir,
                filename=base_name_all + ".png",
                title=f"Layer-wise Activation Rates ({mode}, rel={t_rel}, all layers)",
            )

            # conv only
            base_name_conv = f"layer_activation_rates_{mode}_rel_{tag}_conv_only"
            save_layer_activation_rates(
                layer_conv,
                output_dir=exp_dir,
                filename=base_name_conv + ".csv",
            )
            plot_layer_activation_rates(
                layer_conv,
                output_dir=exp_dir,
                filename=base_name_conv + ".png",
                title=f"Layer-wise Activation Rates ({mode}, rel={t_rel}, conv only)",
            )

        threshold_results[mode] = mode_result

    # --------------------
    # 6. 汇总并保存所有指标
    # --------------------
    all_metrics = {
        "classification": classification_metrics,
        "best_epoch": best_epoch,
        "used_existing_best_model_from_exp1": loaded_from_exp1,
        "config": {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "patience": patience,
            "lr": lr,
            "weight_decay": weight_decay,
            "seed": seed,
            "image_size": image_size,
            "absolute_thresholds": absolute_thresholds,
            "relative_thresholds": relative_thresholds,
            "max_batches_for_activation": max_batches_for_activation,
        },
        "threshold_ablation": threshold_results,
    }

    save_metrics(all_metrics, output_dir=exp_dir, filename="summary_metrics_exp2.json")

    print("[Exp2] Experiment 2 finished.")
    print(f"[Exp2] Results saved under: {exp_dir}")


if __name__ == "__main__":
    import argparse
    from utils import parse_gpu_ids

    parser = argparse.ArgumentParser(description="Experiment 2: Threshold Ablation for Activation Metrics")

    parser.add_argument("--model", type=str, default="resnet18",
                        help="Model name (e.g. resnet34, vgg16)")
    parser.add_argument("--dataset", type=str, default="cifar100",
                        help="Dataset name (currently only cifar100)")
    parser.add_argument("--data-root", type=str, default="./data",
                        help="Root directory for datasets")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Maximum number of training epochs")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight decay")
    parser.add_argument("--gpu-ids", type=str, default="0",
                        help="GPU ids, e.g. '0', '0,1', '-1' for CPU")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--image-size", type=int, default=32,
                        help="Input image size after resize")

    parser.add_argument("--abs-thresholds", type=str, default="0.001,0.005,0.01,0.02,0.05,0.1",
                        help="Comma-separated absolute thresholds, e.g. '0.001,0.005,0.01'")
    parser.add_argument("--rel-thresholds", type=str, default="0.5,1.0,2.0",
                        help="Comma-separated relative thresholds (multipliers), e.g. '0.5,1.0,2.0'")
    parser.add_argument("--max-batches-activation", type=int, default=None,
                        help="Max number of batches used to compute activation metrics (for speed).")
    parser.add_argument("--cifar-stem", type=int, default=1,
                        help="Use CIFAR-style stem for ResNet (1=yes,0=no)")

    args = parser.parse_args()

    def _parse_float_list(s: str) -> List[float]:
        s = s.strip()
        if not s:
            return []
        return [float(x) for x in s.split(",") if x.strip() != ""]

    abs_thresholds = _parse_float_list(args.abs_thresholds)
    rel_thresholds = _parse_float_list(args.rel_thresholds)
    gpu_ids = parse_gpu_ids(args.gpu_ids)

    run_experiment(
        experiment_name="exp2_threshold_ablation",
        model_name=args.model,
        dataset_name=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        gpu_ids=gpu_ids,
        image_size=args.image_size,
        absolute_thresholds=abs_thresholds if abs_thresholds else None,
        relative_thresholds=rel_thresholds if rel_thresholds else None,
        max_batches_for_activation=args.max_batches_activation,
        cifar_stem=bool(args.cifar_stem)
    )
