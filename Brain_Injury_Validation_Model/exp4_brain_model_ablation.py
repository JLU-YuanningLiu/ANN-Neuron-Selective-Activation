"""
Experiment 4: 确定“病脑模型”的最佳实现（DropPath/DropBlock 的位置与参数）

思路：
  - 以 CIFAR-100、可选模型（默认 resnet34）为基线。
  - 统一先获得 baseline 最佳模型（优先复用既有；若缺失则训练一次）。
  - 在“推理阶段”（默认）或“训练阶段”对不同 scope + 参数进行网格搜索：
      * 方法：DropPath / DropBlock
      * 位置 scope：all / stem / early / mid / late
          - ResNet: stem=顶层conv1；early=layer1；mid=layer2&layer3；late=layer4
          - VGG: 将 features 内的 Conv 分为三等分 → early/mid/late；stem=第一个 Conv
      * 参数：
          - DropPath: prob ∈ {0.1, 0.2, 0.3}
          - DropBlock: prob ∈ {0.05, 0.1, 0.2}, block_size ∈ {3, 5, 7}
  - 指标：
      * 分类指标（top-1、top-5 error、loss）
      * 激活指标：权重法（阈值=绝对 0.01），统计 all / conv-only 的全局与分层
  - 结果写入：
      results/exp4_brain_model_ablation/<method>_<phase>_<scope>_p{p}[_b{b}]/
"""

import os
from typing import Dict, List, Optional, Tuple

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
    list_conv2d_names,
    register_drop_path_on_selected_convs,
    register_drop_block_on_selected_convs,
)

# -------------------- 默认网格 --------------------
DEFAULT_DROPPATH_PROBS = [0.1, 0.2, 0.3]
DEFAULT_DROPBLOCK_PROBS = [0.05, 0.1, 0.2]
DEFAULT_DROPBLOCK_SIZES = [3, 5, 7]
DEFAULT_SCOPES = ["all", "stem", "early", "mid", "late"]

# -------------------- Scope 解析 --------------------
def _split_vgg_conv_names(names: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """将 VGG 的 conv 名称按索引等分为 early/mid/late 三段。"""
    n = len(names)
    if n == 0:
        return [], [], []
    cut1 = n // 3
    cut2 = (2 * n) // 3
    early = names[:max(1, cut1)]
    mid = names[max(1, cut1):max(2, cut2)]
    late = names[max(2, cut2):]
    return early, mid, late

def _resolve_scope_selected_names(model: nn.Module, model_name: str, scope: str) -> List[str]:
    """
    根据模型结构与 scope（all/stem/early/mid/late）返回需要施加 Drop 的 Conv 名称列表。
    仅依赖模块名字符串，适配 torchvision.resnet 与 torchvision.vgg。
    """
    conv_names = list_conv2d_names(model)
    scope = scope.lower()

    if scope == "all":
        return conv_names

    # ResNet: stem=顶层 conv1；layer1/2/3/4
    if "resnet" in model_name:
        stem = [n for n in conv_names if n == "conv1"]
        early = [n for n in conv_names if n.startswith("layer1.")]
        mid = [n for n in conv_names if n.startswith("layer2.") or n.startswith("layer3.")]
        late = [n for n in conv_names if n.startswith("layer4.")]
        if scope == "stem":
            return stem
        elif scope == "early":
            return early
        elif scope == "mid":
            return mid
        elif scope == "late":
            return late
        else:
            return []

    # VGG: features.N.conv；按 conv 顺序三等分
    if "vgg" in model_name:
        conv_in_features = [n for n in conv_names if n.startswith("features.")]
        # stem = 第一个 conv
        stem = conv_in_features[:1]
        early, mid, late = _split_vgg_conv_names(conv_in_features)
        if scope == "stem":
            return stem
        elif scope == "early":
            return early
        elif scope == "mid":
            return mid
        elif scope == "late":
            return late
        else:
            return []

    # 其它模型：回退为“全部”
    return conv_names if scope == "stem" else []

# -------------------- 训练辅助 --------------------
def _train_baseline_if_needed(
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
) -> None:
    """确保 baseline 最优模型可用；若不存在则训练一次并保存。"""
    ensure_dir(out_dir)
    tmp_model = build_model(model_name=model_name, num_classes=100, pretrained=False, cifar_stem=True)
    tmp_model = setup_model_on_gpus(tmp_model, gpu_ids=gpu_ids)
    loaded = load_model_if_exists(tmp_model, output_dir=out_dir, filename="best_model.pth", device=device)
    if loaded:
        return
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = SGD(tmp_model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    #scheduler = get_linear_scheduler(optimizer, num_epochs=num_epochs)
    #scheduler = get_cosine_scheduler_with_warmup(optimizer, num_epochs=num_epochs, warmup_epochs=5)
    scheduler = get_warmup_then_multistep(optimizer, num_epochs=num_epochs,
                                                warmup_epochs=5, warmup_start_factor=0.1,
                                                gamma=0.1)
    
    early_stopping = EarlyStopping(patience=patience, min_delta=0.0, min_epochs=161)
    print("[Exp4] Training baseline model...")
    tmp_model, history, best_epoch = train_model(
        model=tmp_model,
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
    plot_training_curves(history, output_dir=out_dir, prefix="exp4_baseline")
    save_model(tmp_model, output_dir=out_dir, filename="best_model.pth")
    print(f"[Exp4] Baseline trained. Best epoch (0-based): {best_epoch}")

def _train_with_train_phase_hooks(
    model: nn.Module,
    method: str,  # "droppath" or "dropblock"
    selected_names: List[str],
    device: torch.device,
    train_loader,
    val_loader,
    num_epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    out_dir: str,
    p: float,
    bsize: Optional[int] = None,
):
    """在训练阶段注册特定 hooks 训练并保存 best。"""
    ensure_dir(out_dir)
    if method == "droppath":
        handles = register_drop_path_on_selected_convs(model, drop_prob=p, selected_names=selected_names, phase="train")
    elif method == "dropblock":
        assert bsize is not None
        handles = register_drop_block_on_selected_convs(model, drop_prob=p, block_size=bsize, selected_names=selected_names, phase="train")
    else:
        raise ValueError(method)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    # scheduler = get_linear_scheduler(optimizer, num_epochs=num_epochs)
    # scheduler = get_cosine_scheduler_with_warmup(optimizer, num_epochs=num_epochs, warmup_epochs=5)
    scheduler = get_warmup_then_multistep(optimizer, num_epochs=num_epochs,
                                          warmup_epochs=5, warmup_start_factor=0.1,
                                          gamma=0.1)

    early_stopping = EarlyStopping(patience=patience, min_delta=0.0, min_epochs=161)

    print(f"[Exp4][train-phase][{method}] Start training with hooks on {len(selected_names)} convs...")
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
    for h in handles: h.remove()

    from visualization import plot_training_curves
    plot_training_curves(history, output_dir=out_dir, prefix="exp4_train")

    save_model(model, output_dir=out_dir, filename="best_model.pth")
    return model, best_epoch

# -------------------- 主实验 --------------------
def run_experiment(
    experiment_name: str = "exp4_brain_model_ablation",
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
    # 搜索空间
    scopes: Optional[List[str]] = None,
    droppath_probs: Optional[List[float]] = None,
    dropblock_probs: Optional[List[float]] = None,
    dropblock_sizes: Optional[List[int]] = None,
    methods: Optional[List[str]] = None,  # ["droppath","dropblock"]
    apply_phase: str = "inference",  # "inference" 或 "train"
    # 激活指标（权重法固定）
    activation_threshold: float = 0.01,
    activation_threshold_type: str = "absolute",
    cifar_stem: bool = True,
):
    if gpu_ids is None: gpu_ids = []
    if scopes is None: scopes = DEFAULT_SCOPES
    if droppath_probs is None: droppath_probs = DEFAULT_DROPPATH_PROBS
    if dropblock_probs is None: dropblock_probs = DEFAULT_DROPBLOCK_PROBS
    if dropblock_sizes is None: dropblock_sizes = DEFAULT_DROPBLOCK_SIZES
    if methods is None: methods = ["droppath", "dropblock"]
    apply_phase = apply_phase.lower()
    assert apply_phase in ("inference", "train")

    # 初始化
    set_random_seeds(seed)
    device = get_device(gpu_ids)
    exp_root = os.path.join("results", experiment_name)
    ensure_dir(exp_root)

    print(f"[Exp4] Experiment directory: {exp_root}")
    print(f"[Exp4] Using device: {device}")
    print(f"[Exp4] Model: {model_name}, Dataset: {dataset_name}")
    print(f"[Exp4] Methods: {methods}, Scopes: {scopes}, Phase: {apply_phase}")

    # 数据
    train_loader, val_loader, test_loader = build_dataloaders(
        dataset_name=dataset_name,
        data_root=data_root,
        batch_size=batch_size,
        num_workers=4,
        val_ratio=0.1,
        image_size=image_size,
    )

    # baseline 目录
    baseline_dir_candidates = [
        os.path.join("results", "exp3_brain_simulation", "baseline"),
        os.path.join("results", "exp1_activation_metrics"),
        os.path.join(exp_root, "baseline"),
    ]
    # 确保至少有一个 baseline
    for cand in baseline_dir_candidates:
        if os.path.exists(os.path.join(cand, "best_model.pth")):
            baseline_dir = cand
            break
    else:
        baseline_dir = baseline_dir_candidates[-1]  # 放到 当前实验目录/baseline
        _train_baseline_if_needed(
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

    overview: Dict[str, Dict] = {}

    # -------------------- 网格搜索 --------------------
    for method in methods:
        if method == "droppath":
            grid = [(p, None) for p in droppath_probs]
        elif method == "dropblock":
            grid = [(p, b) for p in dropblock_probs for b in dropblock_sizes]
        else:
            raise ValueError(f"Unknown method: {method}")

        for scope in scopes:
            # 准备选中层名（通过 baseline 构图即可）
            tmp_model = build_model(model_name=model_name, num_classes=100, pretrained=False, cifar_stem=cifar_stem)
            selected_names = _resolve_scope_selected_names(tmp_model, model_name, scope)

            for p, bsize in grid:
                tag = f"{method}_{apply_phase}_{scope}_p{str(p).replace('.','p')}" + (f"_b{bsize}" if bsize is not None else "")
                out_dir = os.path.join(exp_root, tag)
                ensure_dir(out_dir)
                print(f"\n[Exp4] ==== Config: {tag} ====")
                print(f"[Exp4] Selected convs ({len(selected_names)}): {selected_names[:6]}{' ...' if len(selected_names)>6 else ''}")

                # 准备模型权重
                model = build_model(model_name=model_name, num_classes=100, pretrained=False, cifar_stem=cifar_stem)
                model = setup_model_on_gpus(model, gpu_ids=gpu_ids)

                if apply_phase == "inference":
                    # 复用 baseline 最优模型
                    loaded = False
                    for cand in baseline_dir_candidates:
                        loaded = load_model_if_exists(model, output_dir=cand, filename="best_model.pth", device=device)
                        if loaded:
                            break
                    if not loaded:
                        raise RuntimeError("Baseline best_model.pth not found; unexpected since we ensured training above.")

                    # 推理阶段注册 hook
                    if method == "droppath":
                        handles = register_drop_path_on_selected_convs(model, drop_prob=p, selected_names=selected_names, phase="inference")
                    else:
                        handles = register_drop_block_on_selected_convs(model, drop_prob=p, block_size=bsize, selected_names=selected_names, phase="inference")

                    # 评估分类
                    cls_metrics = evaluate_classification(model, test_loader, device, criterion=nn.CrossEntropyLoss())
                    for h in handles: h.remove()

                    best_epoch = None  # 推理阶段不训练

                else:  # apply_phase == "train"
                    # 若该配置已有 best_model，直接加载；否则训练
                    loaded = load_model_if_exists(model, output_dir=out_dir, filename="best_model.pth", device=device)
                    if not loaded:
                        model, best_epoch = _train_with_train_phase_hooks(
                            model=model,
                            method=method,
                            selected_names=selected_names,
                            device=device,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            num_epochs=num_epochs,
                            patience=patience,
                            lr=lr,
                            weight_decay=weight_decay,
                            out_dir=out_dir,
                            p=p,
                            bsize=bsize,
                        )
                    else:
                        best_epoch = None
                        print("[Exp4] Use existing trained model for this config.")
                    # 评估分类（无 hook）
                    cls_metrics = evaluate_classification(model, test_loader, device, criterion=nn.CrossEntropyLoss())

                print(f"[Exp4] Classification: {cls_metrics}")

                # 权重法激活指标（无论 phase，都是对权重统计）
                (global_all, layer_all, global_conv, layer_conv) = compute_weight_based_activation(
                    model=model,
                    threshold=activation_threshold,
                    threshold_type=activation_threshold_type,
                )

                # 保存汇总
                cfg_metrics = {
                    "classification": cls_metrics,
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
                    "config": {
                        "method": method,
                        "apply_phase": apply_phase,
                        "scope": scope,
                        "p": p,
                        "block_size": bsize,
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
                    "best_epoch": best_epoch,
                    "selected_conv_names": selected_names,
                }
                save_metrics(cfg_metrics, output_dir=out_dir, filename="summary_metrics.json")

                # 分层激活率（CSV+PNG）
                from visualization import plot_layer_activation_rates
                save_layer_activation_rates(layer_all, output_dir=out_dir, filename="layer_activation_rates_weight_all.csv")
                plot_layer_activation_rates(layer_all, output_dir=out_dir, filename="layer_activation_rates_weight_all.png",
                                            title=f"Activation (weight, all) - {tag}")
                save_layer_activation_rates(layer_conv, output_dir=out_dir, filename="layer_activation_rates_weight_conv_only.csv")
                plot_layer_activation_rates(layer_conv, output_dir=out_dir, filename="layer_activation_rates_weight_conv_only.png",
                                            title=f"Activation (weight, conv-only) - {tag}")

                # 概览表
                overview[tag] = {
                    "top1_acc": cls_metrics.get("top1_acc", None),
                    "top5_error": cls_metrics.get("top5_error", None),
                    "activation_all_global": global_all,
                    "activation_conv_global": global_conv,
                }

    save_metrics(overview, output_dir=exp_root, filename="overview.json")
    print("[Exp4] Experiment 4 finished.")
    print(f"[Exp4] Results saved under: {exp_root}")


if __name__ == "__main__":
    import argparse
    from utils import parse_gpu_ids

    parser = argparse.ArgumentParser(description="Experiment 4: Ablation for best sick-brain model (DropPath/DropBlock)")

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

    parser.add_argument("--methods", type=str, default="droppath,dropblock")  # 子集可选
    parser.add_argument("--scopes", type=str, default="all,stem,early,mid,late")
    parser.add_argument("--apply-phase", type=str, default="inference", choices=["inference","train"])

    parser.add_argument("--droppath-probs", type=str, default="0.1,0.2,0.3")
    parser.add_argument("--dropblock-probs", type=str, default="0.05,0.1,0.2")
    parser.add_argument("--dropblock-sizes", type=str, default="3,5,7")

    # 激活（权重法）
    parser.add_argument("--activation-threshold", type=float, default=0.01)
    parser.add_argument("--activation-threshold-type", type=str, default="absolute")
    parser.add_argument("--cifar-stem", type=int, default=1, help="Use CIFAR-style stem for ResNet (1=yes,0=no)")

    args = parser.parse_args()
    gpu_ids = parse_gpu_ids(args.gpu_ids)

    def _parse_floats(s): return [float(x) for x in (s or "").replace("，",",").split(",") if x.strip()]
    def _parse_ints(s):   return [int(x) for x in (s or "").replace("，",",").split(",") if x.strip()]
    def _parse_strs(s):   return [x.strip() for x in (s or "").replace("，",",").split(",") if x.strip()]

    run_experiment(
        experiment_name="exp4_brain_model_ablation",
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
        methods=_parse_strs(args.methods),
        scopes=_parse_strs(args.scopes),
        apply_phase=args.apply_phase,
        droppath_probs=_parse_floats(args.droppath_probs),
        dropblock_probs=_parse_floats(args.dropblock_probs),
        dropblock_sizes=_parse_ints(args.dropblock_sizes),
        activation_threshold=args.activation_threshold,
        activation_threshold_type=args.activation_threshold_type,
        cifar_stem=bool(args.cifar_stem)
    )
