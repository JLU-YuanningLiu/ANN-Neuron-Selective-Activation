"""
Experiment 1: 激活指标设计实验

任务：
    - 基础任务：CIFAR-100 分类。
    - 模型：ResNet-34 / VGG-16（本实验默认使用 ResNet-34，可通过参数切换）。
    - 数据集预处理：Resize + 归一化（使用 CIFAR-100 统计量），后续可扩展。
    - 策略：
        * 多 GPU 选择机制（DataParallel）。
        * 早停策略（基于验证集 loss）。
        * 线性学习率衰减策略（从初始 lr 线性衰减到接近 0）。
    - 激活指标：
        * 基于权重的激活（weight-based）。
        * 基于前向传播的激活（forward-based）。
        * 权重 + 激活联合判定（combined）。
      在本实验中：
        * 模型、数据集、激活阈值保持不变（唯一变量是“激活指标方式”）。
        * 激活阈值使用绝对阈值 0.01。
        * 仅统计 Conv2d 和 Linear 层的激活情况，输出全局激活率和分层激活率。
    - 结果保存：
        * 指标数据：分类准确率、Top-5 错误率、各层激活比率。
        * 图表：训练损失图、训练准确率图、各层激活比率图。
        * 最佳模型权重（按验证集 loss）。
"""

import os
from typing import List, Optional

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
from results import save_layer_activation_rates, save_metrics, save_model, load_model_if_exists
from train import EarlyStopping, get_linear_scheduler, get_cosine_scheduler_with_warmup, get_multistep_scheduler, get_warmup_then_multistep, train_model
from utils import ensure_dir, get_device, set_random_seeds, setup_model_on_gpus


def run_experiment(
    experiment_name: str = "exp1_activation_metrics",
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
    activation_threshold: float = 0.01,
    activation_threshold_type: str = "absolute",
    cifar_stem: bool = True,
):
    """
    Run Experiment 1.

    参数中已经预留：
        - model_name: 用于后续扩展更多模型。
        - dataset_name: 用于后续扩展更多数据集。
        - activation_threshold_type: 目前仅支持 "absolute"，"relative" 接口已保留但未实现。
    """
    if gpu_ids is None:
        gpu_ids = []

    # --------------------
    # 0. 准备工作：随机种子、设备、输出目录
    # --------------------
    set_random_seeds(seed)
    device = get_device(gpu_ids)

    exp_dir = os.path.join("results", experiment_name)
    ensure_dir(exp_dir)

    print(f"[Exp1] Experiment directory: {exp_dir}")
    print(f"[Exp1] Using device: {device}")
    print(f"[Exp1] Model: {model_name}, Dataset: {dataset_name}")
    print(f"[Exp1] Activation threshold: {activation_threshold} ({activation_threshold_type})")

    # --------------------
    # 1. 数据集 & Dataloader
    # --------------------
    print("[Exp1] Building dataloaders...")
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
    print("[Exp1] Building model...")
    # CIFAR-100: num_classes = 100
    num_classes = 100
    model = build_model(model_name=model_name, num_classes=num_classes, pretrained=False, cifar_stem=cifar_stem)
    model = setup_model_on_gpus(model, gpu_ids=gpu_ids)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # --------------------
    # 2.1 若已有 best_model 则跳过训练
    # --------------------
    history = None
    best_epoch = None

    loaded = load_model_if_exists(
        model,
        output_dir=exp_dir,
        filename="best_model.pth",
        device=device,
    )

    if loaded:
        print("[Exp1] Found existing best model. Skip training and use loaded weights.")
    else:

        # --------------------
        # 3. 训练配置：损失函数、优化器、学习率调度、早停
        # --------------------

        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        #scheduler = get_linear_scheduler(optimizer, num_epochs=num_epochs)
        #scheduler = get_cosine_scheduler_with_warmup(optimizer, num_epochs=num_epochs, warmup_epochs=5)
        scheduler = get_warmup_then_multistep(optimizer, num_epochs=num_epochs,
                                                    warmup_epochs=5, warmup_start_factor=0.1,
                                                    gamma=0.1)

        early_stopping = EarlyStopping(patience=patience, min_delta=0.0, min_epochs=161)

        # --------------------
        # 4. 训练
        # --------------------
        print("[Exp1] Start training...")
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
        print(f"[Exp1] Training finished. Best epoch (0-based): {best_epoch}")

        # 保存训练曲线图
        from visualization import plot_training_curves

        plot_training_curves(history, output_dir=exp_dir, prefix="exp1")

        # --------------------
        # 5. 保存最佳模型
        # --------------------
        save_model(model, output_dir=exp_dir, filename="best_model.pth")

    # --------------------
    # 6. 分类性能评估（在测试集上）
    # --------------------
    print("[Exp1] Evaluating classification performance on test set...")
    classification_metrics = evaluate_classification(
        model=model,
        dataloader=test_loader,
        device=device,
        criterion=criterion,
    )
    print(f"[Exp1] Classification metrics: {classification_metrics}")

    # --------------------
    # 7. 激活指标计算（在推理阶段，使用同一个模型和数据集）
    #    激活方式为唯一变量：weight / forward / combined
    # --------------------
    print("[Exp1] Computing activation metrics (weight-based)...")
    (global_weight_all,
     layer_weight_all,
     global_weight_conv,
     layer_weight_conv) = compute_weight_based_activation(
        model=model,
        threshold=activation_threshold,
        threshold_type=activation_threshold_type,
    )

    print("[Exp1] Computing activation metrics (forward-based)...")
    (global_forward_all,
     layer_forward_all,
     global_forward_conv,
     layer_forward_conv) = compute_forward_based_activation(
        model=model,
        dataloader=test_loader,
        device=device,
        threshold=activation_threshold,
        threshold_type=activation_threshold_type,
        max_batches=None,
    )

    print("[Exp1] Computing activation metrics (combined weight + forward)...")
    (global_combined_all,
     layer_combined_all,
     global_combined_conv,
     layer_combined_conv) = compute_combined_weight_forward_activation(
        model=model,
        dataloader=test_loader,
        device=device,
        threshold=activation_threshold,
        threshold_type=activation_threshold_type,
        max_batches=None,
    )

    # --------------------
    # 8. 结果整合与保存
    # --------------------
    # 8.1 指标数据保存（分类 + 全局激活率）
    all_metrics = {
        "classification": classification_metrics,
        "activation": {
            "weight_based": {
                "all_layers": {
                    "global_activation_rate": global_weight_all,
                    "layer_activation_rates": layer_weight_all,
                },
                "conv_only": {
                    "global_activation_rate": global_weight_conv,
                    "layer_activation_rates": layer_weight_conv,
                },
            },
            "forward_based": {
                "all_layers": {
                    "global_activation_rate": global_forward_all,
                    "layer_activation_rates": layer_forward_all,
                },
                "conv_only": {
                    "global_activation_rate": global_forward_conv,
                    "layer_activation_rates": layer_forward_conv,
                },
            },
            "combined": {
                "all_layers": {
                    "global_activation_rate": global_combined_all,
                    "layer_activation_rates": layer_combined_all,
                },
                "conv_only": {
                    "global_activation_rate": global_combined_conv,
                    "layer_activation_rates": layer_combined_conv,
                },
            },
        },
        "best_epoch": best_epoch,
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
            "activation_threshold": activation_threshold,
            "activation_threshold_type": activation_threshold_type,
        },
    }
    save_metrics(all_metrics, output_dir=exp_dir, filename="summary_metrics.json")

    # 8.2 各层激活比率保存（CSV）+ 可视化
    from visualization import plot_layer_activation_rates

    # weight-based: all layers
    save_layer_activation_rates(
        layer_weight_all,
        output_dir=exp_dir,
        filename="layer_activation_rates_weight.csv",
    )
    plot_layer_activation_rates(
        layer_weight_all,
        output_dir=exp_dir,
        filename="layer_activation_rates_weight.png",
        title="Layer-wise Activation Rates (Weight-based, all layers)",
    )

    # weight-based: conv only
    save_layer_activation_rates(
        layer_weight_conv,
        output_dir=exp_dir,
        filename="layer_activation_rates_weight_conv_only.csv",
    )
    plot_layer_activation_rates(
        layer_weight_conv,
        output_dir=exp_dir,
        filename="layer_activation_rates_weight_conv_only.png",
        title="Layer-wise Activation Rates (Weight-based, conv only)",
    )

    # forward-based: all layers
    save_layer_activation_rates(
        layer_forward_all,
        output_dir=exp_dir,
        filename="layer_activation_rates_forward.csv",
    )
    plot_layer_activation_rates(
        layer_forward_all,
        output_dir=exp_dir,
        filename="layer_activation_rates_forward.png",
        title="Layer-wise Activation Rates (Forward-based, all layers)",
    )

    # forward-based: conv only
    save_layer_activation_rates(
        layer_forward_conv,
        output_dir=exp_dir,
        filename="layer_activation_rates_forward_conv_only.csv",
    )
    plot_layer_activation_rates(
        layer_forward_conv,
        output_dir=exp_dir,
        filename="layer_activation_rates_forward_conv_only.png",
        title="Layer-wise Activation Rates (Forward-based, conv only)",
    )

    # combined: all layers
    save_layer_activation_rates(
        layer_combined_all,
        output_dir=exp_dir,
        filename="layer_activation_rates_combined.csv",
    )
    plot_layer_activation_rates(
        layer_combined_all,
        output_dir=exp_dir,
        filename="layer_activation_rates_combined.png",
        title="Layer-wise Activation Rates (Combined, all layers)",
    )

    # combined: conv only
    save_layer_activation_rates(
        layer_combined_conv,
        output_dir=exp_dir,
        filename="layer_activation_rates_combined_conv_only.csv",
    )
    plot_layer_activation_rates(
        layer_combined_conv,
        output_dir=exp_dir,
        filename="layer_activation_rates_combined_conv_only.png",
        title="Layer-wise Activation Rates (Combined, conv only)",
    )

    print("[Exp1] Experiment 1 finished.")
    print(f"[Exp1] Results saved under: {exp_dir}")


if __name__ == "__main__":
    import argparse

    from utils import parse_gpu_ids

    parser = argparse.ArgumentParser(description="Experiment 1: Activation Metric Design on CIFAR-100")

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
    parser.add_argument("--activation-threshold", type=float, default=0.01,
                        help="Activation threshold (absolute)")
    parser.add_argument("--activation-threshold-type", type=str, default="absolute",
                        help="Threshold type: 'absolute' or 'relative' (future use)")
    parser.add_argument("--cifar-stem", type=int, default=1,
                        help="Use CIFAR-style stem for ResNet (1=yes,0=no)")

    args = parser.parse_args()
    gpu_ids = parse_gpu_ids(args.gpu_ids)

    run_experiment(
        experiment_name="exp1_activation_metrics",
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
        activation_threshold=args.activation_threshold,
        activation_threshold_type=args.activation_threshold_type,
        cifar_stem=bool(args.cifar_stem)
    )
