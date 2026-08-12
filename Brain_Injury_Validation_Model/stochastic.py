from typing import List, Iterable, Set, Tuple, Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Stochastic Depth / DropPath 的简化实现。
    对每个样本整条“路径”施加一个按 keep_prob 缩放的 0/1 mask。

    这里我们将它应用在 conv 层的输出特征上：
        - shape: [N, C, H, W] 或 [N, C]
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    # 在 batch 维度上采样 mask，其他维度广播
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    binary_mask = random_tensor.floor()
    return x / keep_prob * binary_mask


def drop_block_2d(
    x: torch.Tensor,
    drop_prob: float = 0.0,
    block_size: int = 3,
    training: bool = False,
) -> torch.Tensor:
    """
    DropPath2D 的简化实现，只用于 conv 特征图 [N, C, H, W]。
    """
    if drop_prob == 0.0 or not training:
        return x

    assert x.dim() == 4, "DropBlock2D expects input of shape [N, C, H, W]"
    N, C, H, W = x.shape

    if H < block_size or W < block_size:
        # 特征图太小就不要做 DropBlock
        return x

    # 计算 gamma, 参考原论文公式
    gamma = drop_prob * H * W / ((block_size ** 2) * (H - block_size + 1) * (W - block_size + 1))

    # 采样中心点 mask: [N, C, H - b + 1, W - b + 1]
    mask = (torch.rand(N, C, H - block_size + 1, W - block_size + 1, device=x.device) < gamma).float()

    # pad 回原尺寸
    pad_h = block_size - 1
    pad_w = block_size - 1
    mask = F.pad(
        mask,
        (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
    )

    # 将中心 mask 展开为 block，得到 [N, C, H, W] 的 block mask
    mask = F.max_pool2d(mask, kernel_size=block_size, stride=1, padding=block_size // 2)
    mask = 1 - mask  # 1 表示保留

    if mask.sum() == 0:
        return x

    # 归一化 scale 保持期望
    return x * mask * (mask.numel() / mask.sum())


def _iter_conv_layers(model: nn.Module):
    """遍历模型中的所有 Conv2d 层"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            yield name, module


def register_drop_path_on_convs(
    model: nn.Module,
    drop_prob: float,
    phase: str = "train",
) -> List[torch.utils.hooks.RemovableHandle]:
    """
    在所有 Conv2d 层输出上注册 DropPath hook。

    phase:
        - "train":  只在 module.training == True 时启用（用于“训练阶段加 DropPath”实验）
        - "inference": 不看 training 标志，总是启用（用于“推理阶段加 DropPath”实验）
    """
    assert phase in ("train", "inference")
    handles = []

    for name, module in _iter_conv_layers(model):

        def hook(mod, inp, out, drop_prob=drop_prob, phase=phase):
            if not isinstance(out, torch.Tensor):
                return out
            if phase == "train":
                training_flag = mod.training
            else:
                training_flag = True  # 推理实验中，即便 eval 模式也强制启用
            return drop_path(out, drop_prob=drop_prob, training=training_flag)

        h = module.register_forward_hook(hook)
        handles.append(h)

    return handles


def register_drop_block_on_convs(
    model: nn.Module,
    drop_prob: float,
    block_size: int,
    phase: str = "train",
) -> List[torch.utils.hooks.RemovableHandle]:
    """
    在所有 Conv2d 层输出上注册 DropBlock2D hook。
    phase 同上。
    """
    assert phase in ("train", "inference")
    handles = []

    for name, module in _iter_conv_layers(model):

        def hook(mod, inp, out, drop_prob=drop_prob, block_size=block_size, phase=phase):
            if not isinstance(out, torch.Tensor):
                return out
            if phase == "train":
                training_flag = mod.training
            else:
                training_flag = True
            return drop_block_2d(
                out,
                drop_prob=drop_prob,
                block_size=block_size,
                training=training_flag,
            )

        h = module.register_forward_hook(hook)
        handles.append(h)

    return handles


def register_dropout_on_convs(
    model: nn.Module,
    drop_prob: float,
    phase: str = "train",
) -> List[torch.utils.hooks.RemovableHandle]:
    """
    在所有 Conv2d 层输出上注册 Dropout hook（对特征图逐元素随机置零）。
    phase:
        - "train":  只在 module.training == True 时启用
        - "inference": 不看 training 标志，总是启用
    """
    assert phase in ("train", "inference")
    handles = []

    for name, module in _iter_conv_layers(model):

        def hook(mod, inp, out, drop_prob=drop_prob, phase=phase):
            if not isinstance(out, torch.Tensor):
                return out
            if phase == "train":
                training_flag = mod.training
            else:
                training_flag = True
            # 与 torch.nn.functional.dropout 保持一致行为
            return F.dropout(out, p=drop_prob, training=training_flag)

        h = module.register_forward_hook(hook)
        handles.append(h)

    return handles


def _name_in_selected(candidate: str, selected: Set[str]) -> bool:
    """兼容 DataParallel 的 'module.' 前缀匹配；也允许精确匹配。"""
    if candidate in selected:
        return True
    if candidate.startswith("module.") and candidate[len("module."):] in selected:
        return True
    return False


def list_conv2d_names(model: nn.Module) -> List[str]:
    """返回模型中所有 Conv2d 层的 name 列表。"""
    names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            names.append(name)
    return names


def register_drop_path_on_selected_convs(
    model: nn.Module,
    drop_prob: float,
    selected_names: Iterable[str],
    phase: str = "train",
) -> List[torch.utils.hooks.RemovableHandle]:
    """仅在 selected_names 指定的 Conv2d 上注册 DropPath（兼容 DataParallel 名称前缀）。"""
    assert phase in ("train", "inference")
    selected: Set[str] = set(selected_names)
    handles = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        if not _name_in_selected(name, selected):
            continue

        def hook(mod, inp, out, drop_prob=drop_prob, phase=phase):
            if not isinstance(out, torch.Tensor):
                return out
            training_flag = (mod.training if phase == "train" else True)
            return drop_path(out, drop_prob=drop_prob, training=training_flag)

        h = module.register_forward_hook(hook)
        handles.append(h)
    return handles


def register_drop_block_on_selected_convs(
    model: nn.Module,
    drop_prob: float,
    block_size: int,
    selected_names: Iterable[str],
    phase: str = "train",
) -> List[torch.utils.hooks.RemovableHandle]:
    """仅在 selected_names 指定的 Conv2d 上注册 DropBlock2D（兼容 DataParallel 名称前缀）。"""
    assert phase in ("train", "inference")
    selected: Set[str] = set(selected_names)
    handles = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        if not _name_in_selected(name, selected):
            continue

        def hook(mod, inp, out, drop_prob=drop_prob, block_size=block_size, phase=phase):
            if not isinstance(out, torch.Tensor):
                return out
            training_flag = (mod.training if phase == "train" else True)
            return drop_block_2d(out, drop_prob=drop_prob, block_size=block_size, training=training_flag)

        h = module.register_forward_hook(hook)
        handles.append(h)
    return handles


def _split_vgg_conv_names(names: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """将 VGG 的 conv 名称按顺序三等分为 early/mid/late。"""
    n = len(names)
    if n == 0:
        return [], [], []
    cut1 = max(1, n // 3)
    cut2 = max(cut1 + 1, (2 * n) // 3)
    early = names[:cut1]
    mid = names[cut1:cut2]
    late = names[cut2:]
    return early, mid, late


def resolve_scope_selected_names(model: nn.Module, model_name: str, scopes: Iterable[str]) -> List[str]:
    """
    根据模型结构与 scopes（如 ['mid','late']）返回需要施加 Drop 的 Conv 名称列表。
    支持：ResNet (conv1, layer1..4) 与 VGG (features.*)；未识别模型回退为 'all'。
    """
    scopes = [s.lower().strip() for s in scopes if s and s.strip()]
    conv_names = list_conv2d_names(model)
    selected: Set[str] = set()

    if "resnet" in model_name.lower():
        stem = [n for n in conv_names if n == "conv1"]
        early = [n for n in conv_names if n.startswith("layer1.")]
        mid = [n for n in conv_names if n.startswith("layer2.") or n.startswith("layer3.")]
        late = [n for n in conv_names if n.startswith("layer4.")]
        mapping = {"stem": stem, "early": early, "mid": mid, "late": late, "all": conv_names}

    elif "vgg" in model_name.lower():
        feats = [n for n in conv_names if n.startswith("features.")]
        stem = feats[:1]
        e, m, l = _split_vgg_conv_names(feats)
        mapping = {"stem": stem, "early": e, "mid": m, "late": l, "all": conv_names}

    else:
        mapping = {"all": conv_names, "stem": [], "early": [], "mid": [], "late": []}

    if not scopes:
        scopes = ["all"]

    for s in scopes:
        selected.update(mapping.get(s, []))
    return sorted(selected)


class EpisodicController:
    """
    用于“时效性”模拟的开关控制器：
      - mode="bernoulli": 每个 batch 以概率 p 进入“异常(开启drop)”状态
      - mode="periodic": 交替 on_k 个 batch 开启，off_k 个 batch 关闭
      - mode="sequence": 按给定 0/1 序列循环
    """
    def __init__(self, mode: str = "bernoulli", p: float = 0.3,
                 on_k: int = 5, off_k: int = 15,
                 sequence: Optional[List[int]] = None, seed: int = 42):
        import random
        self.mode = mode
        self.p = p
        self.on_k, self.off_k = max(1, on_k), max(1, off_k)
        self.sequence = sequence[:] if sequence else None
        self.rng = random.Random(seed)
        self._state = False
        self._countdown = self.on_k  # for periodic
        self._seq_idx = 0

    def next(self) -> bool:
        if self.mode == "bernoulli":
            self._state = self.rng.random() < self.p
        elif self.mode == "periodic":
            if self._countdown <= 0:
                self._state = not self._state
                self._countdown = self.on_k if self._state else self.off_k
            self._countdown -= 1
        elif self.mode == "sequence":
            if not self.sequence:
                self._state = False
            else:
                self._state = bool(self.sequence[self._seq_idx % len(self.sequence)])
                self._seq_idx += 1
        else:
            self._state = False
        return self._state


def register_temporal_drop_path_on_selected_convs(
    model: nn.Module,
    drop_prob: float,
    selected_names: Iterable[str],
    get_enabled: Callable[[], bool],
    phase: str = "inference",
) -> List[torch.utils.hooks.RemovableHandle]:
    """仅在 selected_names 指定 Conv2d 上注册 DropPath；每次前向依据 get_enabled() 决定是否生效。"""
    assert phase in ("train", "inference")
    selected: Set[str] = set(selected_names)
    handles = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        if not _name_in_selected(name, selected):
            continue
        def hook(mod, inp, out, drop_prob=drop_prob, phase=phase, get_enabled=get_enabled):
            if not isinstance(out, torch.Tensor) or not get_enabled():
                return out
            training_flag = (mod.training if phase == "train" else True)
            return drop_path(out, drop_prob=drop_prob, training=training_flag)
        handles.append(module.register_forward_hook(hook))
    return handles


def register_temporal_drop_block_on_selected_convs(
    model: nn.Module,
    drop_prob: float,
    block_size: int,
    selected_names: Iterable[str],
    get_enabled: Callable[[], bool],
    phase: str = "inference",
) -> List[torch.utils.hooks.RemovableHandle]:
    """仅在 selected_names 指定 Conv2d 上注册 DropBlock2D；每次前向依据 get_enabled() 决定是否生效。"""
    assert phase in ("train", "inference")
    selected: Set[str] = set(selected_names)
    handles = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        if not _name_in_selected(name, selected):
            continue
        def hook(mod, inp, out, drop_prob=drop_prob, block_size=block_size, phase=phase, get_enabled=get_enabled):
            if not isinstance(out, torch.Tensor) or not get_enabled():
                return out
            training_flag = (mod.training if phase == "train" else True)
            return drop_block_2d(out, drop_prob=drop_prob, block_size=block_size, training=training_flag)
        handles.append(module.register_forward_hook(hook))
    return handles