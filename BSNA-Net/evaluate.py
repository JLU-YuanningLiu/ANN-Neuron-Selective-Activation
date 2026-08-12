import argparse
import json
import torch
from factory import load_config, build_experiment
from engine.trainer import evaluate
from utils import load_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num-classes", type=int)
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--dense", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    cfg = load_config(args.config)
    model, _, val_loader, task = build_experiment(cfg, args.model, args.data, args.num_classes, args.pretrained, args.workers)
    model.to(args.device)
    load_checkpoint(args.checkpoint, model, map_location=args.device)
    metrics = evaluate(model, val_loader, task, args.device, sparse=not args.dense)
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
