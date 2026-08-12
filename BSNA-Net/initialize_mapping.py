import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from factory import load_config, build_experiment
from engine.mapping import fit_centroids, initialize_neuron_mapping
from utils import set_seed, save_checkpoint, load_checkpoint


def subset_loader(loader, fraction, seed):
    n = len(loader.dataset)
    keep = max(1, int(round(n * fraction)))
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g)[:keep].tolist()
    subset = Subset(loader.dataset, idx)
    return DataLoader(subset, batch_size=loader.batch_size, shuffle=False, num_workers=loader.num_workers, collate_fn=loader.collate_fn, pin_memory=getattr(loader, "pin_memory", False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--num-classes", type=int)
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--activation-threshold", type=float, default=0.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    set_seed(args.seed)
    cfg = load_config(args.config)
    model, train_loader, _, task = build_experiment(cfg, args.model, args.data, args.num_classes, args.pretrained, args.workers, args.seed)
    model.to(args.device)
    load_checkpoint(args.checkpoint, model, map_location=args.device)
    loader = subset_loader(train_loader, cfg.get("mapping_fraction", 0.15), args.seed)
    fit_centroids(model, loader, task, args.device, random_state=args.seed)
    candidates = initialize_neuron_mapping(model, loader, task, args.device, alpha=args.alpha, activation_threshold=args.activation_threshold)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(args.output, model, extra={"config": cfg, "model": args.model})
    summary = {name: float(mask.mean()) for name, mask in candidates.items()}
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
