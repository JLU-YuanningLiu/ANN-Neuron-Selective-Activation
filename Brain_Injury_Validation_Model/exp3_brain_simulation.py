"""
Experiment 3: 用 DropPath / DropBlock 模拟“病脑”的能力与方式

条件：
  - baseline：正常训练
  - train_droppath：训练阶段加 DropPath
  - train_dropblock：训练阶段加 DropBlock
  - train_dropout：训练阶段加 Dropout
  - infer_droppath：推理阶段加 DropPath（复用 baseline 最佳模型）
  - infer_dropblock：推理阶段加 DropBlock（复用 baseline 最佳模型）

统一设置：
  - 任务：CIFAR-100 分类
  - 模型：可选（默认 resnet34）
  - 预处理：Resize + Normalize
  - 训练策略：多 GPU、早停、线性 LR 衰减
  - 激活指标：基于“权重”的激活（阈值 = 绝对 0.01），统计 Conv/Linear；输出全局与分层，另含 conv-only
"""

import os
from typing import List, Optional, Dict

import torch
from torch import nn
from torch.optim import SGD

from datasets import build_dataloaders
from eval import evaluate_classification
from models import build_model
from results import save_layer_activation_rates, save_metrics, save_model, load_model_if_exists
from train import EarlyStopping, get_linear_scheduler, get_cosine_scheduler_with_warmup, get_multistep_scheduler, get_warmup_then_multistep, train_model
from utils import ensure_dir, get_device, set_random_seeds, setup_model_on_gpus
from activations import compute_weight_based_activation
from stochastic import (
    register_drop_path_on_convs,
    register_drop_block_on_convs,
    register_dropout_on_convs,  # 新增的通用函数
)

# -------------------------
# 条件标签
# -------------------------
BASELINE = "baseline"
TRAIN_DROPPATH = "train_droppath"
TRAIN_DROPBLOCK = "train_dropblock"
TRAIN_DROPOUT = "train_dropout"
INFER_DROPPATH = "infer_droppath"
INFER_DROPBLOCK = "infer_dropblock"

DEFAULT_CONDITIONS = [
    BASELINE,
    TRAIN_DROPPATH,
    TRAIN_DROPBLOCK,
    TRAIN_DROPOUT,
    INFER_DROPPATH,
    INFER_DROPBLOCK,
]


def _cond_dir(root: str, cond: str) -> str:
    d = os.path.join(root, cond)
    ensure_dir(d)
    return d


def _train_one_condition(
    cond: str,
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    num_epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    out_dir: str,
    droppath_prob: float,
    dropblock_prob: float,
    dropblock_size: int,
    dropout_p: float,
):
    """
    根据条件在训练阶段注册相应 hook；训练并保存最佳模型与训练曲线。
    """
    # 训练阶段 hooks
    handles = []
    if cond == TRAIN_DROPPATH:
        handles = register_drop_path_on_convs(model, drop_prob=droppath_prob, phase="train")
    elif cond == TRAIN_DROPBLOCK:
        handles = register_drop_block_on_convs(model, drop_prob=dropblock_prob, block_size=dropblock_size, phase="train")
    elif cond == TRAIN_DROPOUT:
        handles = register_dropout_on_convs(model, drop_prob=dropout_p, phase="train")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    #scheduler = get_linear_scheduler(optimizer, num_epochs=num_epochs)
    #scheduler = get_cosine_scheduler_with_warmup(optimizer, num_epochs=num_epochs, warmup_epochs=5)
    scheduler = get_warmup_then_multistep(optimizer, num_epochs=num_epochs,
                                                warmup_epochs=5, warmup_start_factor=0.1,
                                                gamma=0.1)

    early_stopping = EarlyStopping(patience=patience, min_delta=0.0, min_epochs=161)

    print(f"[Exp3][{cond}] Start training...")
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
    print(f"[Exp3][{cond}] Training finished. Best epoch (0-based): {best_epoch}")

    # 移除训练阶段 hooks
    for h in handles:
        h.remove()

    # 训练曲线
    from visualization import plot_training_curves
    plot_training_curves(history, output_dir=out_dir, prefix=f"exp3_{cond}")

    # 保存最佳模型（按 val loss）
    save_model(model, output_dir=out_dir, filename="best_model.pth")

    return model, best_epoch


def _eval_with_optional_infer_hooks(
    cond: str,
    model: nn.Module,
    test_loader,
    device: torch.device,
    droppath_prob: float,
    dropblock_prob: float,
    dropblock_size: int,
):
    """
    对推理阶段改造条件，注册 inference hook 后再评测；其它条件不加 hook。
    """
    handles = []
    if cond == INFER_DROPPATH:
        handles = register_drop_path_on_convs(model, drop_prob=droppath_prob, phase="inference")
    elif cond == INFER_DROPBLOCK:
        handles = register_drop_block_on_convs(model, drop_prob=dropblock_prob, block_size=dropblock_size, phase="inference")

    # 分类指标（top1 / top5 error / loss）
    classification_metrics = evaluate_classification(
        model=model,
        dataloader=test_loader,
        device=device,
        criterion=nn.CrossEntropyLoss(),
    )

    # 移除推理 hooks
    for h in handles:
        h.remove()

    return classification_metrics


def run_experiment(
    experiment_name: str = "exp3_brain_simulation",
    conditions: Optional[List[str]] = None,
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
    # Drop 系列默认参数（“初步设定并固定为一个合理值”）
    droppath_prob: float = 0.2,
    dropblock_prob: float = 0.1,
    dropblock_size: int = 3,
    dropout_p: float = 0.3,
    # 激活指标（基于权重）
    activation_threshold: float = 0.01,           # 绝对阈值
    activation_threshold_type: str = "absolute",  # 接口保留
    cifar_stem: bool = True,
):
    if gpu_ids is None:
        gpu_ids = []
    if conditions is None or len(conditions) == 0 or (len(conditions) == 1 and conditions[0].lower() == "all"):
        conditions = DEFAULT_CONDITIONS

    # 基本准备
    set_random_seeds(seed)
    device = get_device(gpu_ids)
    root_dir = os.path.join("results", experiment_name)
    ensure_dir(root_dir)

    print(f"[Exp3] Experiment directory: {root_dir}")
    print(f"[Exp3] Using device: {device}")
    print(f"[Exp3] Conditions: {conditions}")
    print(f"[Exp3] Model: {model_name}, Dataset: {dataset_name}")
    print(f"[Exp3] Drop settings: droppath={droppath_prob}, dropblock(p={dropblock_prob}, size={dropblock_size}), dropout={dropout_p}")
    print(f"[Exp3] Activation (weight-based): threshold={activation_threshold} ({activation_threshold_type})")

    # 数据
    print("[Exp3] Building dataloaders...")
    train_loader, val_loader, test_loader = build_dataloaders(
        dataset_name=dataset_name,
        data_root=data_root,
        batch_size=batch_size,
        num_workers=4,
        val_ratio=0.1,
        image_size=image_size,
    )

    # baseline 目录（供推理条件复用）
    baseline_dir = _cond_dir(root_dir, BASELINE)

    summary_records: Dict[str, Dict] = {}

    for cond in conditions:
        print(f"\n[Exp3] ===== Condition: {cond} =====")
        cond_dir = _cond_dir(root_dir, cond)

        # 构建模型
        num_classes = 100  # CIFAR-100
        model = build_model(model_name=model_name, num_classes=num_classes, pretrained=False, cifar_stem=cifar_stem)
        model = setup_model_on_gpus(model, gpu_ids=gpu_ids)

        # 选择训练/加载逻辑
        need_training = False
        load_dir = cond_dir

        if cond in (INFER_DROPPATH, INFER_DROPBLOCK):
            # 推理阶段改造：优先复用 baseline 的最优模型
            load_dir = baseline_dir
            loaded = load_model_if_exists(model, output_dir=load_dir, filename="best_model.pth", device=device)
            if not loaded:
                print(f"[Exp3][{cond}] Baseline best model not found. Will train baseline first.")
                # 训练 baseline
                base_model = build_model(model_name=model_name, num_classes=num_classes, pretrained=False, cifar_stem=cifar_stem)
                base_model = setup_model_on_gpus(base_model, gpu_ids=gpu_ids)
                base_loaded = load_model_if_exists(base_model, output_dir=baseline_dir, filename="best_model.pth", device=device)
                if not base_loaded:
                    base_model, _ = _train_one_condition(
                        cond=BASELINE,
                        model=base_model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        device=device,
                        num_epochs=num_epochs,
                        patience=patience,
                        lr=lr,
                        weight_decay=weight_decay,
                        out_dir=baseline_dir,
                        droppath_prob=droppath_prob,
                        dropblock_prob=dropblock_prob,
                        dropblock_size=dropblock_size,
                        dropout_p=dropout_p,
                    )
                # 将 baseline 权重加载到当前模型
                load_model_if_exists(model, output_dir=baseline_dir, filename="best_model.pth", device=device)

        else:
            # 训练阶段改造或 baseline：若本条件最优模型存在则跳过训练，否则训练
            loaded = load_model_if_exists(model, output_dir=cond_dir, filename="best_model.pth", device=device)
            need_training = not loaded

        # 训练（若需要）
        if cond in (BASELINE, TRAIN_DROPPATH, TRAIN_DROPBLOCK, TRAIN_DROPOUT) and need_training:
            model, best_epoch = _train_one_condition(
                cond=cond,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                num_epochs=num_epochs,
                patience=patience,
                lr=lr,
                weight_decay=weight_decay,
                out_dir=cond_dir,
                droppath_prob=droppath_prob,
                dropblock_prob=dropblock_prob,
                dropblock_size=dropblock_size,
                dropout_p=dropout_p,
            )
        else:
            best_epoch = None
            print(f"[Exp3][{cond}] Skip training and use existing best weights from: {load_dir}")

        # ----------------------
        # 评估（分类 & 激活）
        # ----------------------
        # 分类指标：推理条件需要在 hook 下评估
        classification_metrics = _eval_with_optional_infer_hooks(
            cond=cond,
            model=model,
            test_loader=test_loader,
            device=device,
            droppath_prob=droppath_prob,
            dropblock_prob=dropblock_prob,
            dropblock_size=dropblock_size,
        )
        print(f"[Exp3][{cond}] Classification metrics: {classification_metrics}")

        # 激活指标（权重法）
        (global_all,
         layer_all,
         global_conv,
         layer_conv) = compute_weight_based_activation(
            model=model,
            threshold=activation_threshold,
            threshold_type=activation_threshold_type,
        )

        # 保存 JSON 汇总（含全局 & 分层; all / conv-only）
        all_metrics = {
            "classification": classification_metrics,
            "activation_weight_based": {
                "all_layers": {
                    "global_activation_rate": global_all,
                    "layer_activation_rates": layer_all,
                },
                "conv_only": {
                    "global_activation_rate": global_conv,
                    "layer_activation_rates": layer_conv,
                },
            },
            "best_epoch": best_epoch,
            "config": {
                "condition": cond,
                "model_name": model_name,
                "dataset_name": dataset_name,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "patience": patience,
                "lr": lr,
                "weight_decay": weight_decay,
                "seed": seed,
                "image_size": image_size,
                "droppath_prob": droppath_prob,
                "dropblock_prob": dropblock_prob,
                "dropblock_size": dropblock_size,
                "dropout_p": dropout_p,
                "activation_threshold": activation_threshold,
                "activation_threshold_type": activation_threshold_type,
                "used_baseline_dir_for_infer_modes": cond in (INFER_DROPPATH, INFER_DROPBLOCK),
            },
        }
        save_metrics(all_metrics, output_dir=cond_dir, filename="summary_metrics.json")

        # 分层激活率：CSV + PNG（all / conv-only）
        from visualization import plot_layer_activation_rates

        save_layer_activation_rates(layer_all, output_dir=cond_dir, filename="layer_activation_rates_weight_all.csv")
        plot_layer_activation_rates(
            layer_all, output_dir=cond_dir,
            filename="layer_activation_rates_weight_all.png",
            title=f"Layer-wise Activation Rates (weight-based, all) - {cond}"
        )

        save_layer_activation_rates(layer_conv, output_dir=cond_dir, filename="layer_activation_rates_weight_conv_only.csv")
        plot_layer_activation_rates(
            layer_conv, output_dir=cond_dir,
            filename="layer_activation_rates_weight_conv_only.png",
            title=f"Layer-wise Activation Rates (weight-based, conv only) - {cond}"
        )

        # 记录到总表（便于快速比对）
        summary_records[cond] = {
            "classification": classification_metrics,
            "activation_weight_based": {
                "all_layers_global": global_all,
                "conv_only_global": global_conv,
            }
        }

    # 在实验根目录再保存一个总览 JSON
    save_metrics(summary_records, output_dir=root_dir, filename="summary_overview.json")
    print("[Exp3] Experiment 3 finished.")
    print(f"[Exp3] Results saved under: {root_dir}")


if __name__ == "__main__":
    import argparse
    from utils import parse_gpu_ids

    parser = argparse.ArgumentParser(description="Experiment 3: Brain Simulation via DropPath/DropBlock/Dropout")

    parser.add_argument("--conditions", type=str, default="all",
                        help="Comma-separated conditions or 'all'. "
                             "Options: baseline,train_droppath,train_dropblock,train_dropout,infer_droppath,infer_dropblock")

    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--gpu-ids", type=str, default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=32)

    # Drop 系列参数（可改；默认即“固定合理值”）
    parser.add_argument("--droppath-prob", type=float, default=0.2)
    parser.add_argument("--dropblock-prob", type=float, default=0.1)
    parser.add_argument("--dropblock-size", type=int, default=3)
    parser.add_argument("--dropout-p", type=float, default=0.3)

    # 激活（权重法）
    parser.add_argument("--activation-threshold", type=float, default=0.01)
    parser.add_argument("--activation-threshold-type", type=str, default="absolute")
    parser.add_argument("--cifar-stem", type=int, default=1, help="Use CIFAR-style stem for ResNet (1=yes,0=no)")

    args = parser.parse_args()
    gpu_ids = parse_gpu_ids(args.gpu_ids)

    def _parse_list(s: str) -> List[str]:
        s = (s or "").strip()
        if not s or s.lower() == "all":
            return ["all"]
        return [x.strip() for x in s.split(",") if x.strip()]

    run_experiment(
        experiment_name="exp3_brain_simulation",
        conditions=_parse_list(args.conditions),
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
        droppath_prob=args.droppath_prob,
        dropblock_prob=args.dropblock_prob,
        dropblock_size=args.dropblock_size,
        dropout_p=args.dropout_p,
        activation_threshold=args.activation_threshold,
        activation_threshold_type=args.activation_threshold_type,
        cifar_stem=bool(args.cifar_stem)
    )
