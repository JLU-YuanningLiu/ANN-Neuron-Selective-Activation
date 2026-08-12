import os
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from datasets import build_dataloaders
from eval import accuracy_topk  # 用于时效性子集指标
from models import build_model
from results import save_layer_activation_rates, save_metrics, save_model, load_model_if_exists
from train import EarlyStopping, get_linear_scheduler, get_cosine_scheduler_with_warmup, get_multistep_scheduler, get_warmup_then_multistep, train_model
from utils import ensure_dir, get_device, set_random_seeds, setup_model_on_gpus
from activations import compute_weight_based_activation
from stochastic import (
    resolve_scope_selected_names,
    register_drop_path_on_selected_convs,
    register_drop_block_on_selected_convs,
    register_temporal_drop_path_on_selected_convs,
    register_temporal_drop_block_on_selected_convs,
    EpisodicController,
)

# -------------------- 预设：渐进性三档 --------------------
SEVERITY_PRESETS = {
    "mild":     {"dp_scopes": ["mid"],          "dp_p": 0.10, "db_scopes": ["late"], "db_p": 0.05, "db_size": 3},
    "moderate": {"dp_scopes": ["mid", "late"],  "dp_p": 0.20, "db_scopes": ["late"], "db_p": 0.10, "db_size": 5},
    "severe":   {"dp_scopes": ["mid", "late"],  "dp_p": 0.30, "db_scopes": ["late"], "db_p": 0.20, "db_size": 7},
}

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
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    #scheduler = get_linear_scheduler(optimizer, num_epochs=num_epochs)
    #scheduler = get_cosine_scheduler_with_warmup(optimizer, num_epochs=num_epochs, warmup_epochs=5)
    scheduler = get_warmup_then_multistep(optimizer, num_epochs=num_epochs,
                                                warmup_epochs=5, warmup_start_factor=0.1,
                                                gamma=0.1)

    early_stopping = EarlyStopping(patience=patience, min_delta=0.0, min_epochs=161)
    print("[Exp6][baseline] Training...")
    model, history, best_epoch = train_model(
        model=model, train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, optimizer=optimizer, device=device,
        num_epochs=num_epochs, scheduler=scheduler, early_stopping=early_stopping
    )
    from visualization import plot_training_curves
    plot_training_curves(history, output_dir=out_dir, prefix="exp6_baseline")
    save_model(model, output_dir=out_dir, filename="best_model.pth")
    print(f"[Exp6][baseline] Best epoch (0-based): {best_epoch}")

def _load_baseline_model(model_name: str, device: torch.device, gpu_ids: List[int], baseline_dirs: List[str]):
    """按候选目录顺序加载 baseline 的 best_model；返回加载后的模型。"""
    model = build_model(model_name=model_name, num_classes=100, pretrained=False, cifar_stem=True)
    model = setup_model_on_gpus(model, gpu_ids=gpu_ids)
    for cand in baseline_dirs:
        if load_model_if_exists(model, output_dir=cand, filename="best_model.pth", device=device):
            return model
    raise RuntimeError("Baseline best_model.pth not found in any candidate dirs.")

def _eval_and_dump_activation(
    tag: str,
    model: nn.Module,
    test_loader,
    device: torch.device,
    out_dir: str,
    activation_threshold: float,
    activation_threshold_type: str,
    extra_cls_metrics: Optional[Dict[str, float]] = None,
):
    """计算权重法激活（全局+分层；all/conv-only）并落盘，附带分类指标（若提供）。"""
    ensure_dir(out_dir)
    (g_all, l_all, g_conv, l_conv) = compute_weight_based_activation(
        model=model, threshold=activation_threshold, threshold_type=activation_threshold_type
    )
    payload = {
        "activation_weight_based": {
            "all_layers": {"global_activation_rate": g_all, "layer_activation_rates": l_all},
            "conv_only":  {"global_activation_rate": g_conv, "layer_activation_rates": l_conv},
        }
    }
    if extra_cls_metrics is not None:
        payload["classification"] = extra_cls_metrics
    save_metrics(payload, output_dir=out_dir, filename="summary_metrics.json")

    from visualization import plot_layer_activation_rates
    save_layer_activation_rates(l_all,  output_dir=out_dir, filename="layer_activation_rates_weight_all.csv")
    save_layer_activation_rates(l_conv, output_dir=out_dir, filename="layer_activation_rates_weight_conv_only.csv")
    plot_layer_activation_rates(l_all,  output_dir=out_dir, filename="layer_activation_rates_weight_all.png",
                                title=f"Activation (weight, all) - {tag}")
    plot_layer_activation_rates(l_conv, output_dir=out_dir, filename="layer_activation_rates_weight_conv_only.png",
                                title=f"Activation (weight, conv-only) - {tag}")

def _evaluate_classification_overall(model: nn.Module, dataloader, device: torch.device) -> Dict[str, float]:
    """与 eval.evaluate_classification 等价的简版（仅分类总体），用于内部调用以减少依赖。"""
    model.eval()
    total_top1 = 0.0
    total_top5 = 0.0
    total_loss = 0.0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(images)
            batch = images.size(0)
            top1, top5 = accuracy_topk(outputs, targets, topk=(1, 5))
            total_top1 += top1
            total_top5 += top5
            total_loss += criterion(outputs, targets).item() * batch
            total_samples += batch
    if total_samples == 0:
        return {"top1_acc": 0.0, "top5_error": 1.0}
    top1_acc = total_top1 / total_samples
    top5_error = 1.0 - (total_top5 / total_samples)
    return {"top1_acc": top1_acc, "top5_error": top5_error, "loss": total_loss / total_samples}

def _evaluate_with_episodic_hooks(
    model: nn.Module,
    dataloader,
    device: torch.device,
    dp_cfg: Optional[Tuple[List[str], float]] = None,      # (scopes, p)
    db_cfg: Optional[Tuple[List[str], float, int]] = None, # (scopes, p, size)
    controller: EpisodicController = None,
) -> Dict[str, Dict[str, float]]:
    """
    评估“时效性”：同一遍 test，按 batch 切换异常开关，统计 overall / normal / episode 三类分类指标。
    仅在“异常”批次对选定层施加 drop。
    """
    assert controller is not None
    # 解析选择的层名（用未包裹的模板模型）
    template = model.module if hasattr(model, "module") else model
    template = type(template)() if False else template  # 仅占位以强调不改结构
    # 直接基于当前 model 的结构解析（更稳妥）
    model_name = model.__class__.__name__.lower()
    # 为解析 scope，我们需要构建同类模型；这里简单处理：调用 resolve_scope_selected_names 时传入 model_name 字符串
    # 若你的 resolve_scope_selected_names 依赖于 torchvision 命名，建议传 "resnet34"/"vgg16" 等外部配置更可靠。

    # 用外部提供的 scopes 即可，不再解析
    # 注册“动态启用”的 hooks
    state = {"enabled": False}
    def get_enabled(): return state["enabled"]

    handles = []
    if dp_cfg is not None and dp_cfg[1] > 0:
        scopes, p = dp_cfg
        sel = resolve_scope_selected_names(model, model_name, scopes)
        handles += register_temporal_drop_path_on_selected_convs(model, drop_prob=p, selected_names=sel, get_enabled=get_enabled, phase="inference")
    if db_cfg is not None and db_cfg[1] > 0:
        scopes, p, size = db_cfg
        sel = resolve_scope_selected_names(model, model_name, scopes)
        handles += register_temporal_drop_block_on_selected_convs(model, drop_prob=p, block_size=size, selected_names=sel, get_enabled=get_enabled, phase="inference")

    # 统计
    crit = nn.CrossEntropyLoss()
    acc = {
        "overall": {"top1": 0.0, "top5": 0.0, "loss": 0.0, "n": 0},
        "normal":  {"top1": 0.0, "top5": 0.0, "loss": 0.0, "n": 0},
        "episode": {"top1": 0.0, "top5": 0.0, "loss": 0.0, "n": 0},
    }

    model.eval()
    with torch.no_grad():
        for images, targets in dataloader:
            state["enabled"] = controller.next()  # 决定本 batch 是否“异常”
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(images)
            bsz = images.size(0)
            t1, t5 = accuracy_topk(outputs, targets, topk=(1, 5))
            loss = crit(outputs, targets).item() * bsz

            # overall
            acc["overall"]["top1"] += t1; acc["overall"]["top5"] += t5; acc["overall"]["loss"] += loss; acc["overall"]["n"] += bsz
            bucket = "episode" if state["enabled"] else "normal"
            acc[bucket]["top1"] += t1; acc[bucket]["top5"] += t5; acc[bucket]["loss"] += loss; acc[bucket]["n"] += bsz

    # 移除 hooks
    for h in handles: h.remove()

    def finalize(x):
        if x["n"] == 0:
            return {"top1_acc": 0.0, "top5_error": 1.0}
        return {"top1_acc": x["top1"]/x["n"], "top5_error": 1.0 - (x["top5"]/x["n"]), "loss": x["loss"]/x["n"]}

    return {k: finalize(v) for k, v in acc.items()}

def run_experiment(
    experiment_name: str = "exp6_progression_temporality",
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
    # 渐进性：要评估的档位
    severities: Optional[List[str]] = None,  # ["mild","moderate","severe"]
    # 时效性控制（默认 periodic: on 5 / off 15）
    episodic_mode: str = "periodic",  # "periodic" | "bernoulli"
    episodic_p: float = 0.3,          # 仅 bernoulli 生效
    episodic_on_k: int = 5,
    episodic_off_k: int = 15,
    episodic_seed: int = 123,
    # 激活（权重法）
    activation_threshold: float = 0.01,
    activation_threshold_type: str = "absolute",
    cifar_stem: bool = True,
):
    if gpu_ids is None: gpu_ids = []
    if severities is None: severities = ["mild", "moderate", "severe"]

    # 基本准备
    set_random_seeds(seed)
    device = get_device(gpu_ids)
    exp_root = os.path.join("results", experiment_name)
    ensure_dir(exp_root)
    print(f"[Exp6] Dir={exp_root} | Device={device} | Model={model_name} | Dataset={dataset_name}")

    # 数据
    train_loader, val_loader, test_loader = build_dataloaders(
        dataset_name=dataset_name, data_root=data_root, batch_size=batch_size,
        num_workers=4, val_ratio=0.1, image_size=image_size,
    )

    # baseline 目录候选（优先复用之前训练过的）
    baseline_dirs = [
        os.path.join("results", "exp5_normal_vs_sick", "baseline"),
        os.path.join("results", "exp3_brain_simulation", "baseline"),
        os.path.join("results", "exp1_activation_metrics"),
        os.path.join(exp_root, "baseline"),
    ]
    # 确保至少有一个 baseline
    if not any(os.path.exists(os.path.join(d, "best_model.pth")) for d in baseline_dirs):
        _ensure_baseline(model_name, train_loader, val_loader, device, num_epochs, patience, lr, weight_decay, baseline_dirs[-1], gpu_ids)

    # 加载 baseline 权重
    base_model = _load_baseline_model(model_name, device, gpu_ids, baseline_dirs)

    # -------------------- (A) 渐进性：三档严重程度（推理阶段施加） --------------------
    progression_overview = {}
    for sev in severities:
        cfg = SEVERITY_PRESETS[sev]
        out_dir = os.path.join(exp_root, "progression", sev); ensure_dir(out_dir)

        # 准备模型副本
        model = build_model(model_name=model_name, num_classes=100, pretrained=False, cifar_stem=cifar_stem)
        model = setup_model_on_gpus(model, gpu_ids)
        # 加载 baseline 权重
        assert load_model_if_exists(model, output_dir=os.path.dirname(baseline_dirs[0]), filename="best_model.pth", device=device) or \
               any(load_model_if_exists(model, output_dir=d, filename="best_model.pth", device=device) for d in baseline_dirs), \
               "Failed to load baseline weights."

        # 注册“推理阶段”的固定 hooks（始终开启）
        sel_dp = resolve_scope_selected_names(model, model_name, cfg["dp_scopes"])
        sel_db = resolve_scope_selected_names(model, model_name, cfg["db_scopes"])
        hs = []
        if cfg["dp_p"] > 0:
            hs += register_drop_path_on_selected_convs(model, drop_prob=cfg["dp_p"], selected_names=sel_dp, phase="inference")
        if cfg["db_p"] > 0:
            hs += register_drop_block_on_selected_convs(model, drop_prob=cfg["db_p"], block_size=cfg["db_size"], selected_names=sel_db, phase="inference")

        # 分类（总体）
        cls = _evaluate_classification_overall(model, test_loader, device)
        # 移除 hooks
        for h in hs: h.remove()
        # 激活（权重法）
        _eval_and_dump_activation(f"progression-{sev}", model, test_loader, device, out_dir,
                                  activation_threshold, activation_threshold_type,
                                  extra_cls_metrics=cls)
        progression_overview[sev] = cls

    save_metrics(progression_overview, output_dir=os.path.join(exp_root, "progression"), filename="overview.json")

    # -------------------- (B) 时效性：发作-缓解 交替（推理阶段按批次动态开关） --------------------
    temporality_dir = os.path.join(exp_root, "temporality"); ensure_dir(temporality_dir)

    # 使用“中度”作为默认病脑参数（可按需调整）
    cfg = SEVERITY_PRESETS["moderate"]
    dp_cfg = (cfg["dp_scopes"], cfg["dp_p"]) if cfg["dp_p"] > 0 else None
    db_cfg = (cfg["db_scopes"], cfg["db_p"], cfg["db_size"]) if cfg["db_p"] > 0 else None

    # 准备模型（加载 baseline）
    model = build_model(model_name=model_name, num_classes=100, pretrained=False, cifar_stem=cifar_stem)
    model = setup_model_on_gpus(model, gpu_ids)
    assert any(load_model_if_exists(model, output_dir=d, filename="best_model.pth", device=device) for d in baseline_dirs)

    # 构造控制器
    if episodic_mode == "bernoulli":
        controller = EpisodicController(mode="bernoulli", p=episodic_p, seed=episodic_seed)
        epi_tag = f"bernoulli_p{str(episodic_p).replace('.','p')}_seed{episodic_seed}"
    else:
        controller = EpisodicController(mode="periodic", on_k=episodic_on_k, off_k=episodic_off_k, seed=episodic_seed)
        epi_tag = f"periodic_on{episodic_on_k}_off{episodic_off_k}_seed{episodic_seed}"

    # 评估：overall/normal/episode 三组分类
    cls_split = _evaluate_with_episodic_hooks(model, test_loader, device, dp_cfg=dp_cfg, db_cfg=db_cfg, controller=controller)
    save_metrics(cls_split, output_dir=temporality_dir, filename=f"classification_{epi_tag}.json")

    # 额外保存一次（不分拆）的权重法激活（仍与状态无关）
    _eval_and_dump_activation(f"temporality-{epi_tag}", model, test_loader, device, temporality_dir,
                              activation_threshold, activation_threshold_type, extra_cls_metrics=cls_split.get("overall", {}))

    print("[Exp6] Experiment 6 finished.")
    print(f"[Exp6] Results saved under: {exp_root}")


if __name__ == "__main__":
    import argparse
    from utils import parse_gpu_ids

    parser = argparse.ArgumentParser(description="Experiment 6: Progression & Temporality Simulation")

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

    parser.add_argument("--severities", type=str, default="mild,moderate,severe")

    parser.add_argument("--episodic-mode", type=str, default="periodic", choices=["periodic","bernoulli"])
    parser.add_argument("--episodic-p", type=float, default=0.3)
    parser.add_argument("--episodic-on-k", type=int, default=5)
    parser.add_argument("--episodic-off-k", type=int, default=15)
    parser.add_argument("--episodic-seed", type=int, default=123)

    parser.add_argument("--activation-threshold", type=float, default=0.01)
    parser.add_argument("--activation-threshold-type", type=str, default="absolute")
    parser.add_argument("--cifar-stem", type=int, default=1, help="Use CIFAR-style stem for ResNet (1=yes,0=no)")

    args = parser.parse_args()
    gpu_ids = parse_gpu_ids(args.gpu_ids)

    def _parse_list(s: str) -> List[str]:
        return [x.strip() for x in (s or "").replace("，", ",").split(",") if x.strip()]

    run_experiment(
        experiment_name="exp6_progression_temporality",
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
        severities=_parse_list(args.severities),
        episodic_mode=args.episodic_mode,
        episodic_p=args.episodic_p,
        episodic_on_k=args.episodic_on_k,
        episodic_off_k=args.episodic_off_k,
        episodic_seed=args.episodic_seed,
        activation_threshold=args.activation_threshold,
        activation_threshold_type=args.activation_threshold_type,
        cifar_stem=bool(args.cifar_stem)
    )