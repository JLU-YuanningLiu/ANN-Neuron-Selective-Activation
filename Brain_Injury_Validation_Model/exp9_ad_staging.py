"""
Experiment 9: AD Staging (Braak I–II / III–IV / V–VI) with NeuroDegenerativeDrop (NDD)

目标：
- 引入真实 AD 病人内嗅皮质 EC II 层与海马 CA1 区神经元丢失比例（按分期）
- 将 EC II 与 CA1 映射到模型 mid / late（默认 ResNet：mid=layer2+layer3, late=layer4）
- 基于已经训练好的 baseline 模型进行一次“重新训练”（300 epochs），模拟从正常到发病并经历分期
- NDD：不可逆结构损伤随训练进展逐步累积（分段线性）
- 在分期节点（epoch 100 / 200 / 300）进行推理评估，并保存：
    * 当前模型权重
    * 分类指标
    * 激活指标（weight-based, abs threshold=0.01；保留 relative 接口）
    * 结构损伤指标（全局+分层；并保存 ECII(mid)/CA1(late) 的区域损伤率）
    * NDD state（mask）

注意：本实验学习率恒定，不使用 warmup+multistep 调度。
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
from stochastic import resolve_scope_selected_names
from train import train_one_epoch, validate
from utils import ensure_dir, get_device, set_random_seeds, setup_model_on_gpus
from visualization import plot_training_curves, plot_layer_activation_rates


# -----------------------------
# 分期配置（用户给定的“取值”）
# -----------------------------
# epoch 区间（强制 200 epochs）
# 0-100:   Braak I–II
# 100-200: Braak III–IV
# 200-300: Braak V–VI
STAGE_NODES = [
    (100, "Braak_I_II"),
    (200, "Braak_III_IV"),
    (300, "Braak_V_VI"),
]

# 结构（区域）目标损伤率：ECII(mid) 与 CA1(late)
# 这里 damage_rate = neuron_loss_rate（用 channel death 近似 neuron loss）
TARGET_DAMAGE = {
    "Braak_I_II":   {"ECII": 0.05, "CA1": 0.10},
    "Braak_III_IV": {"ECII": 0.35, "CA1": 0.45},
    "Braak_V_VI":   {"ECII": 0.75, "CA1": 0.70},
}

# Aβ + tau 情况 -> 对应 NDD 的两种粒度：
# - path(单通道) ~ tau
# - block(通道组) ~ Aβ
#
# 由于 NDD 的 apply_targets 接口是“同一批 layer dict 使用同一组 path/block component”，
# 我们对 mid 与 late 分开调用，从而允许同一 epoch 内 mid/late 使用不同 component。
STAGE_COMPONENTS = {
    # tau-only：Aβ 少见 -> block 关闭（或极小）
    "Braak_I_II": {
        "ECII": {"path_component": 0.5, "block_component": 0.0},
        "CA1":  {"path_component": 0.2, "block_component": 0.2},
    },
    # ECII：Aβ+tau 共存；CA1：tau 增多，Aβ 开始增多但偏少
    "Braak_III_IV": {
        "ECII": {"path_component": 1.0, "block_component": 1.0},
        "CA1":  {"path_component": 1.0, "block_component": 0.5},
    },
    # 都是 Aβ+tau 共存
    "Braak_V_VI": {
        "ECII": {"path_component": 1.0, "block_component": 1.0},
        "CA1":  {"path_component": 1.0, "block_component": 1.0},
    },
}


def _write_csv_kv(path: str, data: Dict[str, float], header=("key", "value")):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(header))
        for k, v in data.items():
            w.writerow([k, v])


def _piecewise_linear_target(t: int, points: List[Tuple[int, float]]) -> float:
    """
    t: 训练进度（1..num_epochs），表示“第 t 个 epoch 结束时”的目标值
    points: [(t0, v0), (t1, v1), ...]，t 单调递增；t0 通常为 0
    返回：在 t 时刻的线性插值目标值
    """
    if t <= points[0][0]:
        return float(points[0][1])
    for i in range(1, len(points)):
        t0, v0 = points[i - 1]
        t1, v1 = points[i]
        if t <= t1:
            denom = max(1, (t1 - t0))
            alpha = float(t - t0) / float(denom)
            return float(v0 + (v1 - v0) * alpha)
    return float(points[-1][1])


def _stage_name_by_t(t: int) -> str:
    if t <= 50:
        return "Braak_I_II"
    elif t <= 100:
        return "Braak_III_IV"
    else:
        return "Braak_V_VI"


def _ensure_and_load_baseline(
    model: nn.Module,
    device: torch.device,
    baseline_candidates: List[str],
) -> str:
    """
    尝试从多个候选目录加载 baseline best_model.pth。
    返回：实际使用的 baseline 目录（用于记录/对照）。
    若都不存在则 raise。
    """
    for cand in baseline_candidates:
        if load_model_if_exists(model, output_dir=cand, filename="best_model.pth", device=device):
            return cand
    raise RuntimeError(
        "Baseline best_model.pth not found in candidates. "
        "Please run exp5/exp7/exp8 baseline first (or place best_model.pth in an expected folder)."
    )


def _eval_and_dump(
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
    保存：classification + weight-based activation（all/conv-only + global/layer）+ structural_damage（可选）
    并输出 layer activation 图与 CSV。
    """
    ensure_dir(out_dir)

    cls = evaluate_classification(model, test_loader, device, criterion=nn.CrossEntropyLoss())
    (g_all, l_all, g_conv, l_conv) = compute_weight_based_activation(
        model=model, threshold=activation_threshold, threshold_type=activation_threshold_type
    )

    metrics = {
        "classification": cls,
        "activation_weight_based": {
            "all_layers": {"global_activation_rate": g_all, "layer_activation_rates": l_all},
            "conv_only":  {"global_activation_rate": g_conv, "layer_activation_rates": l_conv},
        },
    }
    if structural_damage is not None:
        metrics["structural_damage"] = structural_damage

    save_metrics(metrics, output_dir=out_dir, filename="summary_metrics.json")

    # CSV + 图（activation）
    from results import save_layer_activation_rates
    save_layer_activation_rates(l_all, out_dir, filename="layer_activation_rates_weight_all.csv")
    save_layer_activation_rates(l_conv, out_dir, filename="layer_activation_rates_weight_conv_only.csv")

    plot_layer_activation_rates(l_all, out_dir, filename="layer_activation_rates_weight_all.png",
                                title=f"Activation (weight, all) - {tag}")
    plot_layer_activation_rates(l_conv, out_dir, filename="layer_activation_rates_weight_conv_only.png",
                                title=f"Activation (weight, conv-only) - {tag}")

    return metrics


def run_experiment(
    experiment_name: str = "exp9_ad_staging",
    model_name: str = "resnet18",
    dataset_name: str = "cifar100",
    data_root: str = "./data",
    batch_size: int = 128,
    seed: int = 42,
    gpu_ids: Optional[List[int]] = None,
    image_size: int = 32,
    cifar_stem: bool = True,
    # --- 激活（权重法） ---
    activation_threshold: float = 0.01,
    activation_threshold_type: str = "absolute",
    # --- NDD 参数 ---
    block_group_size: int = 4,
    interval: int = 1,  # 每多少 epoch 更新一次损伤目标（默认每 epoch）
    # --- 重新训练设置 ---
    num_epochs: int = 200,  # 强制 200
    # 恒定学习率：默认取 0.1 经 3 次 *0.1 后的量级 = 1e-4
    retrain_lr: float = 1e-4,
    weight_decay: float = 5e-4,
):
    if gpu_ids is None:
        gpu_ids = []
    num_epochs = 300  # 强制

    set_random_seeds(seed)
    device = get_device(gpu_ids)
    root = os.path.join("results", experiment_name)
    ensure_dir(root)

    # ----------------- 数据 -----------------
    train_loader, val_loader, test_loader = build_dataloaders(
        dataset_name=dataset_name,
        data_root=data_root,
        batch_size=batch_size,
        num_workers=4,
        val_ratio=0.1,
        image_size=image_size,
    )

    # ----------------- 目录结构 -----------------
    baseline_dir = os.path.join(root, "baseline")
    staging_dir = os.path.join(root, "staging_progression")  # 保存 50/100/200 节点的全部结果
    ensure_dir(baseline_dir)
    ensure_dir(staging_dir)

    # 若所有节点 checkpoint 都存在，则跳过整段 retrain
    expected_ckpts = [
        os.path.join(staging_dir, f"epoch_{e:03d}_{name}", "model.pth")
        for (e, name) in STAGE_NODES
    ]
    final_ckpt = os.path.join(root, "final_model.pth")
    if all(os.path.exists(p) for p in expected_ckpts) and os.path.exists(final_ckpt):
        print(f"[Exp9] Found all stage checkpoints + final_model, skip retraining. Root={root}")
        return

    print(f"[Exp9] Root: {root}")
    print(f"[Exp9] Device: {device}")
    print(f"[Exp9] Model: {model_name}, Dataset: {dataset_name}")
    print(f"[Exp9] Retrain epochs={num_epochs}, constant_lr={retrain_lr}, wd={weight_decay}")
    print(f"[Exp9] Activation (weight): threshold={activation_threshold} ({activation_threshold_type})")
    print(f"[Exp9] NDD: block_group_size={block_group_size}, interval={interval}")

    # ----------------- baseline：加载并评估（用于对照 & 用作初始化） -----------------
    base_model = build_model(model_name=model_name, num_classes=100, pretrained=False, cifar_stem=cifar_stem)
    base_model = setup_model_on_gpus(base_model, gpu_ids=gpu_ids)

    baseline_candidates = [
        os.path.join("results", "exp8_neurodegenerative_drop", "baseline"),
        os.path.join("results", "exp7_irreversible_damage", "baseline"),
        os.path.join("results", "exp5_normal_vs_sick", "baseline"),
        baseline_dir,
    ]
    used_baseline_dir = _ensure_and_load_baseline(base_model, device=device, baseline_candidates=baseline_candidates)
    print(f"[Exp9] Loaded baseline from: {used_baseline_dir}")

    # baseline 评估（存到 exp9/baseline）
    baseline_metrics = _eval_and_dump(
        tag="baseline",
        model=base_model,
        test_loader=test_loader,
        device=device,
        out_dir=baseline_dir,
        activation_threshold=activation_threshold,
        activation_threshold_type=activation_threshold_type,
        structural_damage=None,
    )
    # 记录 baseline 来源
    with open(os.path.join(baseline_dir, "baseline_source.json"), "w", encoding="utf-8") as f:
        json.dump({"baseline_dir_used": used_baseline_dir}, f, ensure_ascii=False, indent=2)

    # ----------------- staging retrain：从 baseline 权重开始继续训练一次 -----------------
    # staging_model 初始化为 baseline 权重
    staging_model = build_model(model_name=model_name, num_classes=100, pretrained=False, cifar_stem=cifar_stem)
    staging_model = setup_model_on_gpus(staging_model, gpu_ids=gpu_ids)
    # copy weights
    if hasattr(staging_model, "module") and hasattr(base_model, "module"):
        staging_model.module.load_state_dict(base_model.module.state_dict())
    else:
        staging_model.load_state_dict(base_model.state_dict())

    # 解析 ECII(mid) 与 CA1(late) 对应的 conv layer names（基于 template，不依赖 DataParallel）
    template = build_model(model_name=model_name, num_classes=100, pretrained=False, cifar_stem=cifar_stem)
    ec_layers = resolve_scope_selected_names(template, model_name, scopes=["mid"])
    ca1_layers = resolve_scope_selected_names(template, model_name, scopes=["late"])

    print(f"[Exp9] Region mapping (approx): ECII -> mid layers ({len(ec_layers)} convs); CA1 -> late layers ({len(ca1_layers)} convs)")

    # 初始化 NDD：允许 mid+late 同时启用 path 与 block（你刚改的限定）
    neuro = NeurodegenerativeDrop(
        model=staging_model,
        model_name=model_name,
        path_scopes=["mid", "late"],
        block_scopes=["mid", "late"],
        block_group_size=block_group_size,
        seed=seed,
    )

    # epoch 结束时的目标损伤率（分段线性）
    # t = epoch_idx+1
    ec_points = [
        (0, 0.0),
        (100, TARGET_DAMAGE["Braak_I_II"]["ECII"]),
        (200, TARGET_DAMAGE["Braak_III_IV"]["ECII"]),
        (300, TARGET_DAMAGE["Braak_V_VI"]["ECII"]),
    ]
    ca1_points = [
        (0, 0.0),
        (100, TARGET_DAMAGE["Braak_I_II"]["CA1"]),
        (200, TARGET_DAMAGE["Braak_III_IV"]["CA1"]),
        (300, TARGET_DAMAGE["Braak_V_VI"]["CA1"]),
    ]

    # 恒定 LR
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = SGD(staging_model.parameters(), lr=retrain_lr, momentum=0.9, weight_decay=weight_decay)

    interval = max(1, int(interval))

    history = {
        "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": [],
        "structural_damage": [],  # 全局（mid+late 被管理层）
    }

    # 节点目录
    stage_node_set = set(e for (e, _) in STAGE_NODES)

    for epoch in range(num_epochs):
        tr_l, tr_a = train_one_epoch(staging_model, train_loader, criterion, optimizer, device)
        va_l, va_a = validate(staging_model, val_loader, criterion, device)

        # 恒定 LR（记录用）
        cur_lr = optimizer.param_groups[0]["lr"]

        # epoch 结束：更新目标损伤（影响下一 epoch），并记录结构损伤率
        t = epoch + 1
        stage_name = _stage_name_by_t(t)

        if (epoch % interval) == 0:
            desired_ec = _piecewise_linear_target(t, ec_points)
            desired_ca1 = _piecewise_linear_target(t, ca1_points)

            # region-specific components
            comp_ec = STAGE_COMPONENTS[stage_name]["ECII"]
            comp_ca1 = STAGE_COMPONENTS[stage_name]["CA1"]

            # 1) 推进 ECII(mid) 到 desired_ec
            removed_ec = neuro.apply_targets(
                target_damage_by_layer={name: desired_ec for name in ec_layers},
                path_component=comp_ec["path_component"],
                block_component=comp_ec["block_component"],
                block_group_size=block_group_size,
            )
            # 2) 推进 CA1(late) 到 desired_ca1
            removed_ca1 = neuro.apply_targets(
                target_damage_by_layer={name: desired_ca1 for name in ca1_layers},
                path_component=comp_ca1["path_component"],
                block_component=comp_ca1["block_component"],
                block_group_size=block_group_size,
            )

            total_removed = (sum(removed_ec.values()) if removed_ec else 0) + (sum(removed_ca1.values()) if removed_ca1 else 0)
            print(
                f"[Exp9][NDD] Epoch {t:03d} stage={stage_name} "
                f"desired_damage: ECII={desired_ec:.3f}, CA1={desired_ca1:.3f} "
                f"new_removed_channels={total_removed}"
            )

        g_damage, layer_damage = neuro.get_structural_damage()
        history["train_loss"].append(tr_l)
        history["val_loss"].append(va_l)
        history["train_acc"].append(tr_a)
        history["val_acc"].append(va_a)
        history["lr"].append(cur_lr)
        history["structural_damage"].append(g_damage)

        print(
            f"[Exp9] Epoch [{t}/{num_epochs}] "
            f"loss={tr_l:.4f}/{va_l:.4f} acc={tr_a:.4f}/{va_a:.4f} "
            f"lr={cur_lr:.6f} global_damage={g_damage:.4f} stage={stage_name}"
        )

        # ----------------- 分期节点：推理评估 + 保存模型/指标/结构损伤 -----------------
        if t in stage_node_set:
            node_name = dict(STAGE_NODES)[t]  # e.g. Braak_I_II
            node_dir = os.path.join(staging_dir, f"epoch_{t:03d}_{node_name}")
            ensure_dir(node_dir)

            # 区域损伤率（ECII(mid) / CA1(late)）
            ec_g, ec_layer = neuro.get_structural_damage(layer_names=ec_layers)
            ca1_g, ca1_layer = neuro.get_structural_damage(layer_names=ca1_layers)

            structural = {
                "all_layers": {"global_damage_rate": g_damage, "layer_damage_rates": layer_damage},
                "conv_only": {"global_damage_rate": g_damage, "layer_damage_rates": layer_damage},
                "regions": {
                    "ECII_mid": {"global_damage_rate": ec_g, "layer_damage_rates": ec_layer},
                    "CA1_late": {"global_damage_rate": ca1_g, "layer_damage_rates": ca1_layer},
                },
                "note": "Regions are approximated by model scopes: ECII->mid, CA1->late.",
            }

            # 保存 NDD state（mask 等）
            neuro_state = neuro.state_dict()
            torch.save(neuro_state, os.path.join(node_dir, "neurodrop_state.pth"))

            # 保存结构损伤 CSV + 图（全层）
            _write_csv_kv(os.path.join(node_dir, "layer_structural_damage_all.csv"), layer_damage,
                          header=("layer", "damage_rate"))
            plot_layer_activation_rates(layer_damage, node_dir, filename="layer_structural_damage_all.png",
                                        title=f"Layer-wise Structural Damage (all managed conv) - epoch {t}")

            # 保存结构损伤 CSV + 图（区域）
            _write_csv_kv(os.path.join(node_dir, "layer_structural_damage_ecii_mid.csv"), ec_layer,
                          header=("layer", "damage_rate"))
            _write_csv_kv(os.path.join(node_dir, "layer_structural_damage_ca1_late.csv"), ca1_layer,
                          header=("layer", "damage_rate"))
            plot_layer_activation_rates(ec_layer, node_dir, filename="layer_structural_damage_ecii_mid.png",
                                        title=f"Structural Damage (ECII~mid) - epoch {t}")
            plot_layer_activation_rates(ca1_layer, node_dir, filename="layer_structural_damage_ca1_late.png",
                                        title=f"Structural Damage (CA1~late) - epoch {t}")

            # 推理评估 + 激活率（weight-based）
            _ = _eval_and_dump(
                tag=f"epoch_{t:03d}_{node_name}",
                model=staging_model,
                test_loader=test_loader,
                device=device,
                out_dir=node_dir,
                activation_threshold=activation_threshold,
                activation_threshold_type=activation_threshold_type,
                structural_damage=structural,
            )

            # 保存节点模型
            save_model(staging_model, output_dir=node_dir, filename="model.pth")

            # 保存节点配置（便于追溯）
            node_cfg = {
                "epoch": t,
                "stage": node_name,
                "targets": TARGET_DAMAGE[node_name],
                "components": STAGE_COMPONENTS[node_name],
                "mapping": {"ECII": "mid", "CA1": "late"},
                "block_group_size": block_group_size,
                "interval": interval,
                "constant_lr": retrain_lr,
            }
            with open(os.path.join(node_dir, "node_config.json"), "w", encoding="utf-8") as f:
                json.dump(node_cfg, f, ensure_ascii=False, indent=2)

    # ----------------- 训练结束：保存最终模型 + 曲线 -----------------
    plot_training_curves(history, output_dir=root, prefix="exp9_staging_retrain")
    save_model(staging_model, output_dir=root, filename="final_model.pth")
    torch.save(neuro.state_dict(), os.path.join(root, "final_neurodrop_state.pth"))

    # 保存总览对比（baseline vs final）
    final_cls = evaluate_classification(staging_model, test_loader, device, criterion=nn.CrossEntropyLoss())
    (fg_all, fl_all, fg_conv, fl_conv) = compute_weight_based_activation(
        model=staging_model, threshold=activation_threshold, threshold_type=activation_threshold_type
    )
    final_damage, _final_layer_damage = neuro.get_structural_damage()
    ec_final, _ = neuro.get_structural_damage(layer_names=ec_layers)
    ca1_final, _ = neuro.get_structural_damage(layer_names=ca1_layers)

    overview = {
        "baseline": {
            "top1_acc": baseline_metrics["classification"].get("top1_acc", None),
            "top5_error": baseline_metrics["classification"].get("top5_error", None),
            "activation_all_global": baseline_metrics["activation_weight_based"]["all_layers"]["global_activation_rate"],
            "activation_conv_global": baseline_metrics["activation_weight_based"]["conv_only"]["global_activation_rate"],
        },
        "final": {
            "top1_acc": final_cls.get("top1_acc", None),
            "top5_error": final_cls.get("top5_error", None),
            "activation_all_global": fg_all,
            "activation_conv_global": fg_conv,
            "structural_damage_global": final_damage,
            "structural_damage_regions": {"ECII_mid": ec_final, "CA1_late": ca1_final},
        },
        "config": {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "seed": seed,
            "image_size": image_size,
            "cifar_stem": cifar_stem,
            "activation_threshold": activation_threshold,
            "activation_threshold_type": activation_threshold_type,
            "constant_lr": retrain_lr,
            "weight_decay": weight_decay,
            "block_group_size": block_group_size,
            "interval": interval,
            "stage_nodes": STAGE_NODES,
            "target_damage": TARGET_DAMAGE,
            "stage_components": STAGE_COMPONENTS,
            "baseline_dir_used": used_baseline_dir,
            "region_mapping_note": "ECII mapped to mid scope; CA1 mapped to late scope (approx).",
        },
    }
    save_metrics(overview, output_dir=root, filename="comparison_overview.json")
    print(f"[Exp9] Done. Results under: {root}")


if __name__ == "__main__":
    import argparse
    from utils import parse_gpu_ids

    p = argparse.ArgumentParser("Exp9: AD Staging with NDD (baseline retrain)")

    p.add_argument("--model", type=str, default="resnet18")
    p.add_argument("--dataset", type=str, default="cifar100")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--gpu-ids", type=str, default="0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--cifar-stem", type=int, default=1)

    # retrain
    p.add_argument("--retrain-lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=5e-4)

    # NDD
    p.add_argument("--block-group-size", type=int, default=4)
    p.add_argument("--interval", type=int, default=1)

    # activation
    p.add_argument("--activation-threshold", type=float, default=0.01)
    p.add_argument("--activation-threshold-type", type=str, default="absolute")

    args = p.parse_args()
    gpu_ids = parse_gpu_ids(args.gpu_ids)

    run_experiment(
        experiment_name="exp9_ad_staging",
        model_name=args.model,
        dataset_name=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        seed=args.seed,
        gpu_ids=gpu_ids,
        image_size=args.image_size,
        cifar_stem=bool(args.cifar_stem),
        activation_threshold=args.activation_threshold,
        activation_threshold_type=args.activation_threshold_type,
        block_group_size=args.block_group_size,
        interval=args.interval,
        retrain_lr=args.retrain_lr,
        weight_decay=args.weight_decay,
    )
