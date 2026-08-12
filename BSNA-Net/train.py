import argparse
import json
import math
from pathlib import Path
import torch
from factory import load_config, build_experiment, build_optimizer
from model.bsna import BSNAObjective
from engine.trainer import train_one_epoch, evaluate
from utils import set_seed, save_checkpoint, load_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--stage", choices=["dense", "sparse"], required=True)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--num-classes", type=int)
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    set_seed(args.seed)
    cfg = load_config(args.config)
    model, train_loader, val_loader, task = build_experiment(cfg, args.model, args.data, args.num_classes, args.pretrained, args.workers, args.seed)
    model.to(args.device)
    if args.checkpoint:
        load_checkpoint(args.checkpoint, model, map_location=args.device)
    optimizer = build_optimizer(model, cfg["models"][args.model])
    bsna = cfg["bsna"]
    objective = BSNAObjective(bsna["lambda_sparse"], bsna["lambda_map"], bsna["lambda_div"], bsna["lambda_budget"])
    base_sparse = bsna["lambda_sparse"]
    base_budget = bsna["lambda_budget"]
    total_steps = args.epochs * len(train_loader)
    global_step = 0
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(args.epochs):
        if args.stage == "sparse":
            if task == "vision":
                warm = max(1, bsna.get("sparse_warmup_epochs", 20))
                objective.lambda_sparse = base_sparse * min((epoch + 1) / warm, 1.0)
            else:
                warm_steps = max(1, int(math.ceil(total_steps * bsna.get("warmup_ratio", 0.1))))
                ratio = min((global_step + len(train_loader)) / warm_steps, 1.0)
                objective.lambda_sparse = base_sparse * ratio
                objective.lambda_budget = base_budget * ratio
        metrics = train_one_epoch(model, train_loader, optimizer, objective, task, args.device, sparse=args.stage == "sparse")
        global_step += len(train_loader)
        val = evaluate(model, val_loader, task, args.device, sparse=args.stage == "sparse")
        row = {"epoch": epoch + 1, "train": metrics, "val": val}
        history.append(row)
        print(json.dumps(row))
        save_checkpoint(args.output, model, optimizer, {"history": history, "config": cfg, "model": args.model})


if __name__ == "__main__":
    main()
