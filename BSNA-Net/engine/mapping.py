import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from utils.batch import move_to_device, split_batch, model_encode, model_forward


@torch.no_grad()
def fit_centroids(model, loader, task, device, max_batches=None, random_state=0):
    model.eval()
    features = []
    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        batch = move_to_device(batch, device)
        inputs, _ = split_batch(batch, task)
        z = model_encode(model, inputs)
        features.append(z.detach().cpu())
    x = torch.cat(features, dim=0).numpy()
    km = KMeans(n_clusters=model.num_subsets, random_state=random_state, n_init="auto")
    km.fit(x)
    centroids = torch.tensor(km.cluster_centers_, dtype=model.centroids.dtype, device=device)
    model.set_centroids(F.normalize(centroids, dim=-1))
    return km


def initialize_neuron_mapping(model, loader, task, device, alpha=0.5, activation_threshold=0.0, max_batches=None):
    model.train()
    model.set_stage("mapping")
    stats = {}
    for name, layer in model.named_bsna_layers():
        stats[name] = {
            "freq": torch.zeros(model.num_subsets, layer.out_dim, device=device),
            "contrib": torch.zeros(model.num_subsets, layer.out_dim, device=device),
            "count": torch.zeros(model.num_subsets, device=device)
        }
    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        batch = move_to_device(batch, device)
        inputs, target = split_batch(batch, task)
        model.zero_grad(set_to_none=True)
        logits, aux = model_forward(model, inputs, return_aux=True)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        subset = aux["subset"].detach()
        for name, layer in model.named_bsna_layers():
            act = layer.last_activation
            if act is None or act.grad is None:
                continue
            avec = layer.activation_vector(act.detach())
            gvec = layer.activation_vector(act.grad.detach())
            active = (avec > activation_threshold).to(avec.dtype)
            contribution = (avec * gvec).abs()
            for k in subset.unique(sorted=True):
                idx = subset == k
                if idx.any():
                    kk = int(k.item())
                    stats[name]["freq"][kk] += active[idx].sum(dim=0)
                    stats[name]["contrib"][kk] += contribution[idx].sum(dim=0)
                    stats[name]["count"][kk] += idx.sum()
    candidates = {}
    for name, layer in model.named_bsna_layers():
        count = stats[name]["count"].clamp_min(1.0).unsqueeze(1)
        freq = stats[name]["freq"] / count
        contrib = stats[name]["contrib"] / count
        contrib = contrib / contrib.sum(dim=1, keepdim=True).clamp_min(1e-8)
        importance = alpha * freq + (1.0 - alpha) * contrib
        keep = max(1, int(round(layer.target_ratio * layer.out_dim)))
        mask = torch.zeros_like(importance)
        idx = importance.topk(keep, dim=1).indices
        mask.scatter_(1, idx, 1.0)
        layer.gate.initialize_from_candidates(model.centroids, mask)
        candidates[name] = mask.detach().cpu()
    model.set_stage("sparse")
    return candidates
