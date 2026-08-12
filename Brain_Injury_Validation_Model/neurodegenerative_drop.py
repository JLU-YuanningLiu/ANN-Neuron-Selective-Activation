"""
神经退行性 Drop（Neurodegenerative Drop, NDD）

目标：
- 在“路径级（类似 DropPath 的单通道）”与“块级（类似 DropBlock 的通道组）”两种粒度上，
  对 Conv2d 输出通道进行“不可逆”损伤：一旦通道死亡，后续训练/推理保持死亡。
- 通过参数控制：只启用 path、只启用 block、或二者同时启用。

生物学含义（当前约定）：
- path（单通道粒度）模拟 tau 相关的突触/神经元丢失（更细粒度的连接/神经元退化）
- block（通道组粒度）模拟 Aβ 斑块导致的局部脑区功能丧失（更粗粒度的块状退化）

约束（本版本更新点）：
- path 与 block 都允许作用在模型 mid + late
- 默认不作用在 stem/early（可通过 scopes 选择 mid/late；后续若需要放开到 early/stem 可再改）

实现要点：
- 维护 per-layer 的存活 mask（BoolTensor[out_channels]）
- 将死亡通道的 weight/bias 置 0
- 在 weight/bias 上注册 grad hook，使死亡通道梯度永远为 0（不可逆）
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from stochastic import resolve_scope_selected_names


class NeurodegenerativeDrop:
    """
    神经退行性 Drop 管理器：
    - 仅对指定 scopes 的 Conv2d 层生效
    - 支持 path（单通道）与 block（通道组）两种不可逆损伤
    """

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        path_scopes: Iterable[str] = ("mid", "late"),
        block_scopes: Iterable[str] = ("mid", "late"),
        block_group_size: int = 4,
        seed: int = 42,
    ):
        """
        Args:
            model: 可能是 DataParallel 包裹的模型
            model_name: 例如 "resnet18"
            path_scopes: 期望启用“路径级（单通道）”损伤的 scope（限制在 mid+late）
            block_scopes: 期望启用“块级（通道组）”损伤的 scope（限制在 mid+late）
            block_group_size: block 损伤的通道组大小
            seed: 随机种子（依赖 torch 全局随机性时，外部 set_random_seeds 也会影响）
        """
        self.model = model
        self.base_model = model.module if hasattr(model, "module") else model
        self.model_name = model_name
        self.block_group_size = max(1, int(block_group_size))
        self.seed = int(seed)

        # --- scope 约束：path 与 block 都只允许 mid+late ---
        path_scopes = [s.lower().strip() for s in path_scopes if s and str(s).strip()]
        block_scopes = [s.lower().strip() for s in block_scopes if s and str(s).strip()]

        allowed_scopes = ("mid", "late")
        path_scopes = [s for s in path_scopes if s in allowed_scopes]
        block_scopes = [s for s in block_scopes if s in allowed_scopes]

        # 解析 layer names（基于 base_model）
        self.path_layer_names: List[str] = resolve_scope_selected_names(self.base_model, model_name, path_scopes)
        self.block_layer_names: List[str] = resolve_scope_selected_names(self.base_model, model_name, block_scopes)

        self.path_layer_set = set(self.path_layer_names)
        self.block_layer_set = set(self.block_layer_names)

        # 维护状态
        self.masks: Dict[str, torch.Tensor] = {}     # layer_name -> Bool[out_c] (True alive)
        self.totals: Dict[str, int] = {}             # layer_name -> out_c
        self.modules: Dict[str, nn.Conv2d] = {}      # layer_name -> module

        # 初始化并注册 grad hook
        for name, module in self.base_model.named_modules():
            if not isinstance(module, nn.Conv2d):
                continue
            if (name not in self.path_layer_set) and (name not in self.block_layer_set):
                continue

            out_c = module.out_channels
            device = module.weight.device
            mask = torch.ones(out_c, dtype=torch.bool, device=device)

            self.masks[name] = mask
            self.totals[name] = out_c
            self.modules[name] = module

            # 权重 grad hook：死亡通道梯度永远为 0
            def _weight_grad_hook(grad, mask_ref=mask):
                if grad is None:
                    return None
                view_shape = (mask_ref.shape[0],) + (1,) * (grad.dim() - 1)
                return grad * mask_ref.view(view_shape).to(grad.device)

            module.weight.register_hook(_weight_grad_hook)

            # bias grad hook
            if module.bias is not None:
                def _bias_grad_hook(grad, mask_ref=mask):
                    if grad is None:
                        return None
                    return grad * mask_ref.to(grad.device)
                module.bias.register_hook(_bias_grad_hook)

        print(
            f"[NeuroDrop] Prepared on {len(self.masks)} conv layers. "
            f"path_layers={len(self.path_layer_set)}, block_layers={len(self.block_layer_set)}"
        )

    # -----------------------------
    # 基础：结构损伤统计/导入导出
    # -----------------------------

    def state_dict(self) -> Dict:
        """导出不可逆损伤状态（仅 masks / totals / layer sets），用于后续恢复训练（保持不可逆）。"""
        payload = {
            "model_name": self.model_name,
            "block_group_size": self.block_group_size,
            "path_layer_names": sorted(list(self.path_layer_set)),
            "block_layer_names": sorted(list(self.block_layer_set)),
            "masks": {k: v.detach().bool().cpu() for k, v in self.masks.items()},
            "totals": dict(self.totals),
        }
        return payload

    def load_state_dict(self, payload: Dict, strict: bool = True) -> None:
        """
        恢复 masks（会把死亡通道重新置 0）。
        注意：grad hook 在 __init__ 已注册，load 只负责同步 mask 与权重清零。
        """
        masks_in = payload.get("masks", {})
        if not isinstance(masks_in, dict):
            raise ValueError("Invalid payload: masks must be a dict.")

        for name, mask_cpu in masks_in.items():
            if name not in self.masks:
                if strict:
                    raise KeyError(f"Layer '{name}' not found in current NeuroDrop manager.")
                continue
            mask = self.masks[name]
            new_mask = mask_cpu.to(mask.device).bool()
            if new_mask.numel() != mask.numel():
                raise ValueError(f"Mask size mismatch for layer '{name}'.")
            mask.copy_(new_mask)

            # 同步清零
            module = self.modules[name]
            with torch.no_grad():
                module.weight.data[~mask] = 0
                if module.bias is not None:
                    module.bias.data[~mask] = 0

    def get_structural_damage(
        self,
        layer_names: Optional[List[str]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Returns:
            global_damage, per_layer_damage
        """
        total_channels = 0
        total_removed = 0
        per_layer: Dict[str, float] = {}

        for name, mask in self.masks.items():
            if layer_names is not None and name not in layer_names:
                continue
            total = int(self.totals[name])
            alive = int(mask.sum().item())
            removed = total - alive
            total_channels += total
            total_removed += removed
            per_layer[name] = float(removed) / float(total) if total > 0 else 0.0

        global_damage = float(total_removed) / float(total_channels) if total_channels > 0 else 0.0
        return global_damage, per_layer

    # -----------------------------
    # 核心：不可逆损伤施加
    # -----------------------------

    def _apply_prune_for_layer(
        self,
        layer_name: str,
        target_damage: float,
        path_component: float,
        block_component: float,
        block_group_size: Optional[int] = None,
    ) -> int:
        """
        将某一层的损伤推进到 target_damage（若当前已达到则不动）。
        返回本次新剪掉的通道数。
        """
        if layer_name not in self.masks:
            return 0
        mask = self.masks[layer_name]
        module = self.modules[layer_name]

        total = int(mask.numel())
        alive_idx = torch.nonzero(mask, as_tuple=True)[0]
        alive = int(alive_idx.numel())
        if alive <= 1:
            return 0

        # clamp target
        target_damage = float(max(0.0, min(1.0, target_damage)))
        # 至少留 1 个通道
        desired_alive = max(1, int(round(total * (1.0 - target_damage))))
        if desired_alive >= alive:
            return 0

        need_kill = alive - desired_alive
        if need_kill <= 0:
            return 0

        allow_path = (layer_name in self.path_layer_set)
        allow_block = (layer_name in self.block_layer_set)

        pc = float(max(0.0, path_component)) if allow_path else 0.0
        bc = float(max(0.0, block_component)) if allow_block else 0.0
        if pc <= 0.0 and bc <= 0.0:
            return 0

        if block_group_size is None:
            block_group_size = self.block_group_size
        block_group_size = max(1, int(block_group_size))

        to_kill = torch.zeros_like(mask, dtype=torch.bool)

        # ---- block（通道组）优先：按比例分配需要 kill 的数量（不允许 overshoot）----
        block_kill = 0
        if bc > 0.0 and block_group_size > 1 and alive >= 2 * block_group_size:
            frac_block = bc / (pc + bc)
            desired_block_kill = int(round(need_kill * frac_block))

            num_groups = alive // block_group_size
            # 至少保留 1 组
            max_groups_to_kill = max(0, num_groups - 1)
            groups_to_kill = min(max_groups_to_kill, desired_block_kill // block_group_size)

            if groups_to_kill > 0:
                perm = torch.randperm(num_groups, device=alive_idx.device)
                chosen_groups = perm[:groups_to_kill]

                group_indices: List[int] = []
                for g in chosen_groups.tolist():
                    start = g * block_group_size
                    end = start + block_group_size
                    if end <= alive:
                        group_indices.extend(range(start, end))

                if group_indices:
                    gi = torch.tensor(group_indices, device=alive_idx.device, dtype=torch.long)
                    kill_idx = alive_idx[gi]
                    to_kill[kill_idx] = True
                    block_kill = int(to_kill.sum().item())

        # ---- path（单通道）补齐剩余 ----
        remaining_need = need_kill - block_kill
        if remaining_need > 0 and pc > 0.0:
            remaining_alive_idx = torch.nonzero(mask & (~to_kill), as_tuple=True)[0]
            remaining_alive = int(remaining_alive_idx.numel())
            if remaining_alive > 1:
                # 保证至少留 1 个
                remaining_need = min(remaining_need, remaining_alive - 1)
                if remaining_need > 0:
                    perm = torch.randperm(remaining_alive, device=remaining_alive_idx.device)
                    chosen = perm[:remaining_need]
                    kill_idx = remaining_alive_idx[chosen]
                    to_kill[kill_idx] = True

        new_removed = int(to_kill.sum().item())
        if new_removed <= 0:
            return 0

        # 应用：mask 更新 + 权重置零（不可逆）
        mask[to_kill] = False
        with torch.no_grad():
            module.weight.data[~mask] = 0
            if module.bias is not None:
                module.bias.data[~mask] = 0

        return new_removed

    def apply_targets(
        self,
        target_damage_by_layer: Dict[str, float],
        path_component: float = 1.0,
        block_component: float = 1.0,
        block_group_size: Optional[int] = None,
    ) -> Dict[str, int]:
        """
        将各层推进到指定 target_damage（按 layer 提供）。
        返回：每层本次新增剪掉的通道数。
        """
        removed: Dict[str, int] = {}
        for name, td in target_damage_by_layer.items():
            r = self._apply_prune_for_layer(
                layer_name=name,
                target_damage=float(td),
                path_component=path_component,
                block_component=block_component,
                block_group_size=block_group_size,
            )
            if r > 0:
                removed[name] = r
        return removed

    def apply_uniform_target(
        self,
        target_damage: float,
        path_component: float = 1.0,
        block_component: float = 1.0,
        block_group_size: Optional[int] = None,
    ) -> Dict[str, int]:
        """将所有被管理层推进到同一个 target_damage（常用于实验8）。"""
        return self.apply_targets(
            target_damage_by_layer={name: target_damage for name in self.masks.keys()},
            path_component=path_component,
            block_component=block_component,
            block_group_size=block_group_size,
        )
