"""
Main entry file.

功能：
    - 选择要运行的实验（目前实现：exp1_activation_metrics）。
    - 选择模型（resnet34 / vgg16 等）。
    - 选择数据集（目前实现：cifar100）。
"""

import argparse

from utils import parse_gpu_ids


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment Runner Main")

    parser.add_argument("--experiment", type=str, default="exp1_activation_metrics",
                        help="Which experiment to run. e.g. exp1_activation_metrics")

    parser.add_argument("--model", type=str, default="resnet18",
                        help="Model name, e.g. resnet18, resnet34, vgg16")

    parser.add_argument("--dataset", type=str, default="cifar100",
                        help="Dataset name, e.g. cifar100")

    parser.add_argument("--data-root", type=str, default="./data",
                        help="Root directory for datasets")

    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size")

    parser.add_argument("--epochs", type=int, default=200,
                        help="Maximum number of training epochs")

    parser.add_argument("--patience", type=int, default=40,
                        help="Early stopping patience")

    parser.add_argument("--lr", type=float, default=0.1,
                        help="Initial learning rate")

    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight decay")

    parser.add_argument("--gpu-ids", type=str, default="0",
                        help="GPU ids, e.g. '0', '0,1', '-1' for CPU")

    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    parser.add_argument("--image-size", type=int, default=32,
                        help="Input image size after resize")

    parser.add_argument("--cifar-stem", type=int, default=1,
                        help="Use CIFAR-style stem for ResNet (1=yes,0=no)")

    parser.add_argument("--activation-threshold", type=float, default=0.01,
                        help="Activation threshold (absolute)")

    parser.add_argument("--activation-threshold-type", type=str, default="absolute",
                        help="Threshold type: 'absolute' or 'relative' (future use)")

    parser.add_argument("--abs-thresholds", type=str, default="0.001,0.005,0.01,0.02,0.05",
                        help="Comma-separated absolute thresholds for threshold ablation (exp2).")

    parser.add_argument("--rel-thresholds", type=str, default="0.5,1.0,2.0",
                        help="Comma-separated relative thresholds for threshold ablation (exp2).")

    parser.add_argument("--max-batches-activation", type=int, default=None,
                        help="Max batches for activation computation in exp2.")

    return parser.parse_args()


def main():
    args = parse_args()
    gpu_ids = parse_gpu_ids(args.gpu_ids)

    def _parse_float_list(s: str):
        s = s.strip()
        if not s:
            return []
        return [float(x) for x in s.split(",") if x.strip() != ""]

    if args.experiment == "exp1_activation_metrics":
        from exp1_activation_metrics import run_experiment

        run_experiment(
            experiment_name="exp1_activation_metrics",
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
            activation_threshold=args.activation_threshold,
            activation_threshold_type=args.activation_threshold_type,
            cifar_stem=bool(args.cifar_stem),
        )

    elif args.experiment == "exp2_threshold_ablation":
        from exp2_threshold_ablation import run_experiment

        abs_thresholds = _parse_float_list(args.abs_thresholds)
        rel_thresholds = _parse_float_list(args.rel_thresholds)

        run_experiment(
            experiment_name="exp2_threshold_ablation",
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
            absolute_thresholds=abs_thresholds if abs_thresholds else None,
            relative_thresholds=rel_thresholds if rel_thresholds else None,
            max_batches_for_activation=args.max_batches_activation,
            cifar_stem=bool(args.cifar_stem),
        )

    elif args.experiment == "exp3_brain_simulation":
        from exp3_brain_simulation import run_experiment

        run_experiment(
            experiment_name="exp3_brain_simulation",
            # 允许默认跑全部条件；也可以之后在 main.py 里透出 --conditions
            conditions=["all"],
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
            # 激活阈值（基于权重法）
            activation_threshold=args.activation_threshold,
            activation_threshold_type=args.activation_threshold_type,
            cifar_stem=bool(args.cifar_stem),
        )

    elif args.experiment == "exp4_brain_model_ablation":
        from exp4_brain_model_ablation import run_experiment
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
            cifar_stem=bool(args.cifar_stem),
        )

    elif args.experiment == "exp5_normal_vs_sick":
        from exp5_normal_vs_sick import run_experiment
        run_experiment(
            experiment_name="exp5_normal_vs_sick",
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
            activation_threshold=args.activation_threshold,
            activation_threshold_type=args.activation_threshold_type,
            cifar_stem=bool(args.cifar_stem),
        )

    elif args.experiment == "exp6_progression_temporality":
        from exp6_progression_temporality import run_experiment
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
            activation_threshold=args.activation_threshold,
            activation_threshold_type=args.activation_threshold_type,
            cifar_stem=bool(args.cifar_stem),
        )

    elif args.experiment == "exp7_irreversible_damage":
        from exp7_irreversible_damage import run_experiment
        run_experiment(
            experiment_name="exp7_irreversible_damage",
            model_name=args.model,
            dataset_name=args.dataset,
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
            gpu_ids=gpu_ids,
            image_size=args.image_size,
            lesion_scopes=[x.strip() for x in args.lesion_scopes.replace("，", ",").split(",") if x.strip()]
            if hasattr(args, "lesion_scopes") else ["mid", "late"],
            lesion_start_fraction=getattr(args, "lesion_start_fraction", 0.3),
            lesion_interval=getattr(args, "lesion_interval", 1),
            lesion_rate_per_event=getattr(args, "lesion_rate_per_event", 0.02),
            activation_threshold=getattr(args, "activation_threshold", 0.01),
            activation_threshold_type=getattr(args, "activation_threshold_type", "absolute"),
            cifar_stem=bool(getattr(args, "cifar_stem", 1)),
        )

    elif args.experiment == "exp8_neurodegenerative_drop":
        from exp8_neurodegenerative_drop import run_experiment
        run_experiment(
            experiment_name="exp8_neurodegenerative_drop",
            model_name=args.model,
            dataset_name=args.dataset,
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
            gpu_ids=gpu_ids,
            image_size=args.image_size,
            cifar_stem=bool(args.cifar_stem),
            activation_threshold=args.activation_threshold,
            activation_threshold_type=args.activation_threshold_type,
        )

    elif args.experiment == "exp9_ad_staging":
        from exp9_ad_staging import run_experiment
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
        )



    else:
        raise  ValueError(f"Unknown experiment: {args.experiment}")


if __name__ == "__main__":
    main()
