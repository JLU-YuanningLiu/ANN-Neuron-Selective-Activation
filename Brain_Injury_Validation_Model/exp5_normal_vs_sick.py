"""
Experiment 5: 正常脑 vs 病脑 对比

设定（基于前序结论的临时“最佳病脑”方案）：
  - 仅在训练阶段加入 Drop：DropPath 作用于 mid+late，DropBlock 作用于 late
  - 权重法激活，阈值=绝对 0.01（接口保留 relative）
  - 其它训练/数据/评估与 Experiment 1 保持一致
"""

import os
from typing import Dict, List, Optional

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
    register_drop_path_on_selected_convs,
    register_drop_block_on_selected_convs,
    resolve_scope_selected_names,
)

BASELINE = "baseline"
SICK = "sick"

def _ensure_baseline(
    model_name: str,
    train_loader,
    val_loader,
    device: torch.device,
    num_epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    out_dir: str,
    gpu_ids: List[int],
):
    """确保 baseline best_model 可用；不存在则训练一次。"""
    ensure_dir(out_dir)
    model = build_model(model_name=model_name, num_classes=100, pretrained=False, cifar_stem=True)
    model = setup_model_on_gpus(model, gpu_ids=gpu_ids)
    if load_model_if_exists(model, output_dir=out_dir, filename="best_model.pth", device=device):
        return
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    #scheduler = get_linear_scheduler(optimizer, num_epochs=num_epochs)
    #scheduler = get_cosine_scheduler_with_warmup(optimizer, num_epochs=num_epochs, warmup_epochs=5)
    scheduler = get_warmup_then_multistep(optimizer, num_epochs=num_epochs,
                                                warmup_epochs=5, warmup_start_factor=0.1,
                                                gamma=0.1)

    early_stopping = EarlyStopping(patience=patience, min_delta=0.0, min_epochs=161)
    print("[Exp5][baseline] Training...")
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
    from visualization import plot_training_curves
    plot_training_curves(history, output_dir=out_dir, prefix="exp5_baseline")
    save_model(model, output_dir=out_dir, filename="best_model.pth")
    print(f"[Exp5][baseline] Best epoch (0-based): {best_epoch}")

def _train_sick_model(
    model: nn.Module,
    model_name: str,
    droppath_scopes: List[str],
    dropblock_scopes: List[str],
    droppath_prob: float,
    dropblock_prob: float,
    dropblock_size: int,
    train_loader,
    val_loader,
    device: torch.device,
    num_epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    out_dir: str,
):
    """在训练阶段施加 DropPath/DropBlock（按给定 scopes），训练并保存最优模型。"""
    ensure_dir(out_dir)

    # 基于未包裹的“模板模型”解析 scope 所对应的 conv 名称
    template = build_model(model_name=model_name, num_classes=100, pretrained=False, cifar_stem=True)
    sel_dp = resolve_scope_selected_names(template, model_name, droppath_scopes)
    sel_db = resolve_scope_selected_names(template, model_name, dropblock_scopes)

    # 在训练模型上注册 hooks（DataParallel 兼容由注册函数内部处理）
    handles = []
    if droppath_prob > 0 and sel_dp:
        handles += register_drop_path_on_selected_convs(model, drop_prob=droppath_prob, selected_names=sel_dp, phase="train")
    if dropblock_prob > 0 and sel_db:
        handles += register_drop_block_on_selected_convs(model, drop_prob=dropblock_prob, block_size=dropblock_size, selected_names=sel_db, phase="train")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    # scheduler = get_linear_scheduler(optimizer, num_epochs=num_epochs)
    # scheduler = get_cosine_scheduler_with_warmup(optimizer, num_epochs=num_epochs, warmup_epochs=5)
    scheduler = get_warmup_then_multistep(optimizer, num_epochs=num_epochs,
                                          warmup_epochs=5, warmup_start_factor=0.1,
                                          gamma=0.1)

    early_stopping = EarlyStopping(patience=patience, min_delta=0.0, min_epochs=161)

    print(f"[Exp5][sick] Training with DropPath scopes={droppath_scopes} p={droppath_prob}; "
          f"DropBlock scopes={dropblock_scopes} p={dropblock_prob}, size={dropblock_size}...")
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

    # 移除 hooks
    for h in handles:
        h.remove()

    from visualization import plot_training_curves
    plot_training_curves(history, output_dir=out_dir, prefix="exp5_sick")
    save_model(model, output_dir=out_dir, filename="best_model.pth")
    print(f"[Exp5][sick] Best epoch (0-based): {best_epoch}")

    return best_epoch

def _eval_and_dump_all(
    tag: str,
    model: nn.Module,
    test_loader,
    device: torch.device,
    out_dir: str,
    activation_threshold: float,
    activation_threshold_type: str,
) -> Dict:
    """评估分类与权重法激活（含全局+分层、all/conv-only），并保存到 out_dir。"""
    ensure_dir(out_dir)
    criterion = nn.CrossEntropyLoss()
    cls = evaluate_classification(model, test_loader, device, criterion=criterion)
    (g_all, l_all, g_conv, l_conv) = compute_weight_based_activation(
        model=model,
        threshold=activation_threshold,
        threshold_type=activation_threshold_type,
    )
    metrics = {
        "classification": cls,
        "activation_weight_based": {
            "all_layers": {"global_activation_rate": g_all, "layer_activation_rates": l_all},
            "conv_only":  {"global_activation_rate": g_conv, "layer_activation_rates": l_conv},
        },
    }
    save_metrics(metrics, output_dir=out_dir, filename="summary_metrics.json")

    from visualization import plot_layer_activation_rates
    save_layer_activation_rates(l_all,  output_dir=out_dir, filename="layer_activation_rates_weight_all.csv")
    save_layer_activation_rates(l_conv, output_dir=out_dir, filename="layer_activation_rates_weight_conv_only.csv")
    plot_layer_activation_rates(l_all,  output_dir=out_dir, filename="layer_activation_rates_weight_all.png",
                                title=f"Activation (weight, all) - {tag}")
    plot_layer_activation_rates(l_conv, output_dir=out_dir, filename="layer_activation_rates_weight_conv_only.png",
                                title=f"Activation (weight, conv-only) - {tag}")
    return metrics


def _dump_weight_activation_as_infer_drop(
    tag: str,
    metrics: Dict,
    out_dir: str,
):
    """
    说明：权重法激活仅依赖于权重，与推理期是否开启 DropPath/DropBlock 无关。
    本函数将已计算的权重法激活结果（all / conv-only，含全局+分层）
    以“infer_drop”标签另存一份，方便对比与归档。
    """
    ensure_dir(out_dir)

    if "activation_weight_based" not in metrics:
        raise ValueError("metrics must contain 'activation_weight_based' from _eval_and_dump_all().")

    aw = metrics["activation_weight_based"]
    l_all  = aw["all_layers"]["layer_activation_rates"]
    l_conv = aw["conv_only"]["layer_activation_rates"]

    # 1) 保存 JSON（带注释）
    payload = {
        "activation_weight_based_with_infer_drop": aw,
        "note": "Weight-based activation depends only on weights; enabling DropPath/DropBlock at inference does not change these values.",
    }
    save_metrics(payload, output_dir=out_dir, filename="summary_activation_infer_drop_weight.json")

    # 2) 保存分层 CSV + 可视化
    from visualization import plot_layer_activation_rates
    save_layer_activation_rates(l_all,  output_dir=out_dir, filename="layer_activation_rates_weight_all_layers_infer_drop.csv")
    save_layer_activation_rates(l_conv, output_dir=out_dir, filename="layer_activation_rates_weight_conv_only_infer_drop.csv")

    plot_layer_activation_rates(
        l_all,  output_dir=out_dir,
        filename="layer_activation_rates_weight_all_layers_infer_drop.png",
        title=f"Weight-based Activation (ALL) - {tag} (inference drop: same as no-drop)"
    )
    plot_layer_activation_rates(
        l_conv, output_dir=out_dir,
        filename="layer_activation_rates_weight_conv_only_infer_drop.png",
        title=f"Weight-based Activation (Conv-only) - {tag} (inference drop: same as no-drop)"
    )


def run_experiment(
    experiment_name: str = "exp5_normal_vs_sick",
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
    # “最佳病脑”设定（可通过 CLI 调整）
    droppath_scopes: Optional[List[str]] = None,  # 默认 ["mid","late"]
    dropblock_scopes: Optional[List[str]] = None, # 默认 ["late"]
    droppath_prob: float = 0.2,
    dropblock_prob: float = 0.1,
    dropblock_size: int = 3,
    # 激活（权重法）
    activation_threshold: float = 0.01,
    activation_threshold_type: str = "absolute",
    cifar_stem: bool = True,
):
    if gpu_ids is None:
        gpu_ids = []
    if droppath_scopes is None:
        droppath_scopes = ["mid", "late"]
    if dropblock_scopes is None:
        dropblock_scopes = ["late"]

    # 基本准备
    set_random_seeds(seed)
    device = get_device(gpu_ids)
    exp_root = os.path.join("results", experiment_name)
    ensure_dir(exp_root)

    print(f"[Exp5] Experiment directory: {exp_root}")
    print(f"[Exp5] Using device: {device}")
    print(f"[Exp5] Model: {model_name}, Dataset: {dataset_name}")
    print(f"[Exp5] Sick config: DropPath scopes={droppath_scopes} p={droppath_prob}; "
          f"DropBlock scopes={dropblock_scopes} p={dropblock_prob} size={dropblock_size}")
    print(f"[Exp5] Activation (weight-based): threshold={activation_threshold} ({activation_threshold_type})")

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
    baseline_dir = os.path.join(exp_root, BASELINE)
    sick_dir = os.path.join(exp_root, SICK)
    ensure_dir(baseline_dir); ensure_dir(sick_dir)

    # 1) baseline：如无则训练
    _ensure_baseline(
        model_name=model_name,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=num_epochs,
        patience=patience,
        lr=lr,
        weight_decay=weight_decay,
        out_dir=baseline_dir,
        gpu_ids=gpu_ids,
    )

    # 2) sick：如无则训练（训练阶段施加 DP/DB）
    sick_model = build_model(model_name=model_name, num_classes=100, pretrained=False, cifar_stem=cifar_stem)
    sick_model = setup_model_on_gpus(sick_model, gpu_ids=gpu_ids)
    if not load_model_if_exists(sick_model, output_dir=sick_dir, filename="best_model.pth", device=device):
        _ = _train_sick_model(
            model=sick_model,
            model_name=model_name,
            droppath_scopes=droppath_scopes,
            dropblock_scopes=dropblock_scopes,
            droppath_prob=droppath_prob,
            dropblock_prob=dropblock_prob,
            dropblock_size=dropblock_size,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=num_epochs,
            patience=patience,
            lr=lr,
            weight_decay=weight_decay,
            out_dir=sick_dir,
        )

    # 3) 载入 baseline 模型（用于评估）
    base_model = build_model(model_name=model_name, num_classes=100, pretrained=False, cifar_stem=cifar_stem)
    base_model = setup_model_on_gpus(base_model, gpu_ids=gpu_ids)
    loaded = load_model_if_exists(base_model, output_dir=baseline_dir, filename="best_model.pth", device=device)
    if not loaded:
        raise RuntimeError("Baseline best_model.pth not found after ensure.")

    # 4) 评估与激活（无 hooks，均为纯推理）
    base_metrics = _eval_and_dump_all(
        tag="baseline",
        model=base_model,
        test_loader=test_loader,
        device=device,
        out_dir=baseline_dir,
        activation_threshold=activation_threshold,
        activation_threshold_type=activation_threshold_type,
    )

    sick_metrics = _eval_and_dump_all(
        tag="sick",
        model=sick_model,
        test_loader=test_loader,
        device=device,
        out_dir=sick_dir,
        activation_threshold=activation_threshold,
        activation_threshold_type=activation_threshold_type,
    )

    # 额外归档：病脑在“推理期开 Drop”的权重法激活率（数值与不开 Drop 相同，这里仅加标签另存）
    infer_drop_weight_dir = os.path.join(sick_dir, "infer_drop_weight")
    _dump_weight_activation_as_infer_drop(
        tag="sick-infer-drop-weight",
        metrics=sick_metrics,
        out_dir=infer_drop_weight_dir,
    )

    # 5) 总览对比
    overview = {
        "baseline": {
            "top1_acc": base_metrics["classification"].get("top1_acc", None),
            "top5_error": base_metrics["classification"].get("top5_error", None),
            "activation_all_global": base_metrics["activation_weight_based"]["all_layers"]["global_activation_rate"],
            "activation_conv_global": base_metrics["activation_weight_based"]["conv_only"]["global_activation_rate"],
        },
        "sick": {
            "top1_acc": sick_metrics["classification"].get("top1_acc", None),
            "top5_error": sick_metrics["classification"].get("top5_error", None),
            "activation_all_global": sick_metrics["activation_weight_based"]["all_layers"]["global_activation_rate"],
            "activation_conv_global": sick_metrics["activation_weight_based"]["conv_only"]["global_activation_rate"],
        },
        "config": {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "droppath_scopes": droppath_scopes,
            "dropblock_scopes": dropblock_scopes,
            "droppath_prob": droppath_prob,
            "dropblock_prob": dropblock_prob,
            "dropblock_size": dropblock_size,
            "activation_threshold": activation_threshold,
            "activation_threshold_type": activation_threshold_type,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "patience": patience,
            "lr": lr,
            "weight_decay": weight_decay,
            "seed": seed,
            "image_size": image_size,
        },
    }
    save_metrics(overview, output_dir=exp_root, filename="comparison_overview.json")
    print("[Exp5] Experiment 5 finished.")
    print(f"[Exp5] Results saved under: {exp_root}")


if __name__ == "__main__":
    import argparse
    from utils import parse_gpu_ids

    parser = argparse.ArgumentParser(description="Experiment 5: Normal vs Sick Brain Comparison")

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

    # 病脑设定（可改；默认“mid+late”与“late”）
    parser.add_argument("--droppath-scopes", type=str, default="mid,late")
    parser.add_argument("--dropblock-scopes", type=str, default="late")
    parser.add_argument("--droppath-prob", type=float, default=0.2)
    parser.add_argument("--dropblock-prob", type=float, default=0.1)
    parser.add_argument("--dropblock-size", type=int, default=3)

    # 激活（权重法）
    parser.add_argument("--activation-threshold", type=float, default=0.01)
    parser.add_argument("--activation-threshold-type", type=str, default="absolute")
    parser.add_argument("--cifar-stem", type=int, default=1, help="Use CIFAR-style stem for ResNet (1=yes,0=no)")

    args = parser.parse_args()
    gpu_ids = parse_gpu_ids(args.gpu_ids)

    def _parse_scopes(s: str) -> List[str]:
        return [x.strip() for x in (s or "").replace("，", ",").split(",") if x.strip()]

    run_experiment(
        experiment_name="exp5_normal_vs_sick",
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
        droppath_scopes=_parse_scopes(args.droppath_scopes),
        dropblock_scopes=_parse_scopes(args.dropblock_scopes),
        droppath_prob=args.droppath_prob,
        dropblock_prob=args.dropblock_prob,
        dropblock_size=args.dropblock_size,
        activation_threshold=args.activation_threshold,
        activation_threshold_type=args.activation_threshold_type,
        cifar_stem=bool(args.cifar_stem)
    )