from contextlib import contextmanager
from contextvars import ContextVar
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

_context = ContextVar("bsna_context", default=None)


class SurrogateStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        a = x.abs()
        h = torch.zeros_like(x)
        h = torch.where(a <= 0.4, 2.0 - 4.0 * a, h)
        h = torch.where((a > 0.4) & (a <= 1.0), torch.full_like(x, 0.4), h)
        return grad_output * h


def surrogate_step(x):
    return SurrogateStep.apply(x)


@contextmanager
def bsna_scope(stage, z=None, centroid=None, subset=None):
    token = _context.set({"stage": stage, "z": z, "centroid": centroid, "subset": subset})
    try:
        yield
    finally:
        _context.reset(token)


class NeuronMappingGate(nn.Module):
    def __init__(self, z_dim, out_dim, num_subsets, eta_init=0.5):
        super().__init__()
        self.z_dim = z_dim
        self.out_dim = out_dim
        self.num_subsets = num_subsets
        self.proj = nn.Linear(z_dim * 2, out_dim)
        self.eta = nn.Parameter(torch.full((out_dim,), eta_init))
        self.register_buffer("candidate_masks", torch.ones(num_subsets, out_dim))
        self.last_gate = None
        self.last_mask = None

    def forward(self, z, centroid, subset):
        g = torch.sigmoid(self.proj(torch.cat([z, centroid], dim=-1)))
        m = surrogate_step(g - self.eta.unsqueeze(0))
        self.last_gate = g
        self.last_mask = m
        return m

    @torch.no_grad()
    def initialize_from_candidates(self, centroids, candidate_masks, logit_scale=2.0):
        self.candidate_masks.copy_(candidate_masks.to(self.candidate_masks.device))
        c = centroids.to(self.proj.weight.device)
        y = torch.where(candidate_masks.to(c.device) > 0, torch.full_like(candidate_masks.to(c.device), logit_scale), torch.full_like(candidate_masks.to(c.device), -logit_scale))
        x = torch.cat([c, torch.ones(c.size(0), 1, device=c.device, dtype=c.dtype)], dim=1)
        solution = torch.linalg.lstsq(x, y).solution
        self.proj.weight.zero_()
        self.proj.weight[:, self.z_dim:] = solution[:-1].T
        self.proj.bias.copy_(solution[-1])


class BSNALayerBase(nn.Module):
    def __init__(self, z_dim, out_dim, num_subsets, target_ratio):
        super().__init__()
        self.out_dim = out_dim
        self.target_ratio = float(target_ratio)
        self.gate = NeuronMappingGate(z_dim, out_dim, num_subsets)
        self.last_activation = None
        self.last_neuron_mask = None

    def _record(self, y):
        self.last_activation = y
        if y.requires_grad:
            y.retain_grad()

    def _apply_gate(self, y, axis):
        ctx = _context.get()
        if ctx is None or ctx["stage"] in {"dense", "mapping"}:
            self.last_neuron_mask = torch.ones(y.size(0), self.out_dim, device=y.device, dtype=y.dtype)
            return y
        m = self.gate(ctx["z"], ctx["centroid"], ctx["subset"])
        self.last_neuron_mask = m
        if axis == "channel":
            shape = [m.size(0), m.size(1)] + [1] * (y.dim() - 2)
            return y * m.view(*shape)
        shape = [m.size(0)] + [1] * (y.dim() - 2) + [m.size(1)]
        return y * m.view(*shape)

    def activation_vector(self, x):
        raise NotImplementedError


class BSNAConv2d(BSNALayerBase):
    def __init__(self, conv, z_dim, num_subsets, target_ratio, threshold_init=0.0):
        super().__init__(z_dim, conv.out_channels, num_subsets, target_ratio)
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.padding_mode = conv.padding_mode
        self.weight = nn.Parameter(conv.weight.detach().clone())
        self.bias = nn.Parameter(conv.bias.detach().clone()) if conv.bias is not None else None
        self.connection_threshold = nn.Parameter(torch.full((conv.out_channels,), threshold_init, dtype=conv.weight.dtype))

    def forward(self, x):
        ctx = _context.get()
        if ctx is None or ctx["stage"] in {"dense", "mapping"}:
            w = self.weight
        else:
            t = self.connection_threshold.view(-1, 1, 1, 1)
            w = self.weight * surrogate_step(self.weight.abs() - t)
        y = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self._record(y)
        return self._apply_gate(y, "channel")

    def activation_vector(self, x):
        if x.dim() <= 2:
            return x
        return x.abs().mean(dim=tuple(range(2, x.dim())))


class BSNAConv1d(BSNALayerBase):
    def __init__(self, conv, z_dim, num_subsets, target_ratio, threshold_init=0.0):
        super().__init__(z_dim, conv.out_channels, num_subsets, target_ratio)
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.padding_mode = conv.padding_mode
        self.weight = nn.Parameter(conv.weight.detach().clone())
        self.bias = nn.Parameter(conv.bias.detach().clone()) if conv.bias is not None else None
        self.connection_threshold = nn.Parameter(torch.full((conv.out_channels,), threshold_init, dtype=conv.weight.dtype))

    def forward(self, x):
        ctx = _context.get()
        if ctx is None or ctx["stage"] in {"dense", "mapping"}:
            w = self.weight
        else:
            t = self.connection_threshold.view(-1, 1, 1)
            w = self.weight * surrogate_step(self.weight.abs() - t)
        y = F.conv1d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self._record(y)
        return self._apply_gate(y, "channel")

    def activation_vector(self, x):
        if x.dim() <= 2:
            return x
        return x.abs().mean(dim=tuple(range(2, x.dim())))


class BSNALinear(BSNALayerBase):
    def __init__(self, linear, z_dim, num_subsets, target_ratio, threshold_init=0.0):
        super().__init__(z_dim, linear.out_features, num_subsets, target_ratio)
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = nn.Parameter(linear.weight.detach().clone())
        self.bias = nn.Parameter(linear.bias.detach().clone()) if linear.bias is not None else None
        self.connection_threshold = nn.Parameter(torch.full((linear.out_features,), threshold_init, dtype=linear.weight.dtype))

    def forward(self, x):
        ctx = _context.get()
        if ctx is None or ctx["stage"] in {"dense", "mapping"}:
            w = self.weight
        else:
            t = self.connection_threshold.view(-1, 1)
            w = self.weight * surrogate_step(self.weight.abs() - t)
        y = F.linear(x, w, self.bias)
        self._record(y)
        return self._apply_gate(y, "last")

    def activation_vector(self, x):
        if x.dim() == 2:
            return x.abs()
        dims = tuple(range(1, x.dim() - 1))
        return x.abs().mean(dim=dims)


class BSNAFeatureGate(BSNALayerBase):
    def __init__(self, dim, z_dim, num_subsets, target_ratio, axis="last"):
        super().__init__(z_dim, dim, num_subsets, target_ratio)
        self.axis = axis

    def forward(self, x):
        self._record(x)
        return self._apply_gate(x, self.axis)

    def activation_vector(self, x):
        if self.axis == "channel":
            if x.dim() <= 2:
                return x.abs()
            return x.abs().mean(dim=tuple(range(2, x.dim())))
        if x.dim() == 2:
            return x.abs()
        return x.abs().mean(dim=tuple(range(1, x.dim() - 1)))


class VisionSemanticEncoder(nn.Module):
    def __init__(self, z_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.proj = nn.Linear(64, z_dim)

    def forward(self, x):
        return F.normalize(self.proj(self.net(x).flatten(1)), dim=-1)


class BSNAClassifierBase(nn.Module):
    def __init__(self, num_subsets, z_dim):
        super().__init__()
        self.num_subsets = num_subsets
        self.z_dim = z_dim
        self.register_buffer("centroids", torch.zeros(num_subsets, z_dim))
        self.register_buffer("centroids_ready", torch.tensor(False))
        self.stage = "dense"

    def set_stage(self, stage):
        if stage not in {"dense", "mapping", "sparse"}:
            raise ValueError(stage)
        self.stage = stage

    @torch.no_grad()
    def set_centroids(self, centroids):
        if centroids.shape != self.centroids.shape:
            raise ValueError((centroids.shape, self.centroids.shape))
        self.centroids.copy_(centroids.to(self.centroids.device, self.centroids.dtype))
        self.centroids_ready.fill_(True)

    def assign_subset(self, z):
        if not bool(self.centroids_ready):
            return torch.zeros(z.size(0), dtype=torch.long, device=z.device)
        d = torch.cdist(z, self.centroids)
        return d.argmin(dim=1)

    def routing(self, z):
        subset = self.assign_subset(z)
        centroid = self.centroids[subset] if bool(self.centroids_ready) else torch.zeros_like(z)
        return subset, centroid

    def named_bsna_layers(self):
        for name, module in self.named_modules():
            if isinstance(module, (BSNAConv1d, BSNAConv2d, BSNALinear, BSNAFeatureGate)):
                yield name, module

    def aux_state(self, z, subset):
        masks = {}
        gates = {}
        for name, layer in self.named_bsna_layers():
            if layer.last_neuron_mask is not None:
                masks[name] = layer.last_neuron_mask
            if layer.gate.last_gate is not None:
                gates[name] = layer.gate.last_gate
        return {"z": z, "subset": subset, "masks": masks, "gates": gates}


class BSNAObjective(nn.Module):
    def __init__(self, lambda_sparse=1e-4, lambda_map=1e-2, lambda_div=5e-3, lambda_budget=5e-3):
        super().__init__()
        self.lambda_sparse = lambda_sparse
        self.lambda_map = lambda_map
        self.lambda_div = lambda_div
        self.lambda_budget = lambda_budget

    def regularizers(self, model, subset):
        device = subset.device
        sparse = torch.zeros((), device=device)
        map_loss = torch.zeros((), device=device)
        div = torch.zeros((), device=device)
        budget = torch.zeros((), device=device)
        for _, layer in model.named_bsna_layers():
            if hasattr(layer, "connection_threshold"):
                sparse = sparse + torch.exp(-layer.connection_threshold).sum()
            m = layer.last_neuron_mask
            if m is None:
                continue
            budget = budget + (m.mean() - layer.target_ratio).pow(2)
            means = []
            for k in subset.unique(sorted=True):
                idx = subset == k
                if idx.any():
                    mk = m[idx].mean(dim=0)
                    means.append(mk)
                    map_loss = map_loss + (m[idx] - mk.unsqueeze(0)).pow(2).mean()
            if len(means) > 1:
                stack = torch.stack(means)
                norm = F.normalize(stack, dim=-1, eps=1e-8)
                sim = norm @ norm.T
                n = sim.size(0)
                div = div + (sim.sum() - sim.diag().sum()) / max(n * (n - 1), 1)
        return {"sparse": sparse, "map": map_loss, "div": div, "budget": budget}

    def forward(self, logits, target, model, subset):
        task = F.cross_entropy(logits, target)
        regs = self.regularizers(model, subset)
        total = task + self.lambda_sparse * regs["sparse"] + self.lambda_map * regs["map"] + self.lambda_div * regs["div"] + self.lambda_budget * regs["budget"]
        return {"loss": total, "task": task, **regs}


def _ratio_for(index, total, name, ratios):
    lname = name.lower()
    if any(x in lname for x in ["classifier", "head", "fc"]):
        return ratios["head"]
    r = (index + 1) / max(total, 1)
    if r <= 1 / 3:
        return ratios["shallow"]
    if r <= 2 / 3:
        return ratios["middle"]
    return ratios["deep"]


def instrument_modules(root, z_dim, num_subsets, selector, ratios):
    selected = []
    for name, module in root.named_modules():
        if name and selector(name, module):
            selected.append((name, module))
    total = len(selected)
    for index, (name, module) in enumerate(selected):
        parent = root
        parts = name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        attr = parts[-1]
        ratio = _ratio_for(index, total, name, ratios)
        if isinstance(module, nn.Conv2d):
            new_module = BSNAConv2d(module, z_dim, num_subsets, ratio)
        elif isinstance(module, nn.Conv1d):
            new_module = BSNAConv1d(module, z_dim, num_subsets, ratio)
        elif isinstance(module, nn.Linear):
            new_module = BSNALinear(module, z_dim, num_subsets, ratio)
        else:
            continue
        setattr(parent, attr, new_module)
    return root


def default_ratios(domain):
    if domain == "vision":
        return {"shallow": 0.15, "middle": 0.08, "deep": 0.04, "head": 0.10}
    return {"shallow": 0.20, "middle": 0.10, "deep": 0.05, "head": 0.12}
