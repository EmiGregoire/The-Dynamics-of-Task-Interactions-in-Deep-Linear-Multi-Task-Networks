from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from network_experiment_utils import run_network_only_experiment


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset-backed MTL runs."""
    parser = argparse.ArgumentParser(
        description=(
            "Run a dataset-backed multitask network experiment and save the results."
        )
    )
    parser.add_argument(
        "--tasktype",
        required=True,
        choices=[
            "multimnist",
            "nyuv2_depth_normal",
            "nyuv2_segmentation_depth",
            "nyuv2_segmentation_normal",
        ],
        help="Dataset/task configuration to run.",
    )
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=1000)
    parser.add_argument("--training-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--model-type", choices=["linear", "relu"], default="linear")
    parser.add_argument("--init-scale", type=float, default=1e-4)
    parser.add_argument(
        "--init-mode",
        choices=["random", "balanced"],
        default="random",
        help="Weight initialization scheme for the shared and task heads.",
    )
    parser.add_argument(
        "--init-lambda",
        type=float,
        default=0.0,
        help="Lambda value used by the balanced initialization scheme.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", choices=["auto", "numpy", "torch"], default="auto")
    parser.add_argument(
    "--save-weight-history",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Store WS/W1/W2 snapshots at each training step.",
)

    parser.add_argument(
        "--prefer-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer the PyTorch CUDA backend when available.",
    )
    parser.add_argument(
        "--trackgradients",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store gradient cosine similarity and gradient magnitudes.",
    )
    parser.add_argument(
        "--weighting-mode",
        choices=["none", "gradnorm", "cossim"],
        default="none",
        help="Dynamic task-loss weighting rule.",
    )
    parser.add_argument("--gradient-surgery", choices=["none", "pcgrad"], default="none")
    parser.add_argument("--main-task", type=int, choices=[1, 2], default=1)
    parser.add_argument("--gradnorm-alpha", type=float, default=1.5)
    parser.add_argument("--gradnorm-lr", type=float, default=0.025)
    parser.add_argument(
        "--make-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save a metrics plot next to the NPZ result file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cluster_runs"),
        help="Directory where outputs will be written.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="run",
        help="Filename stem for saved outputs.",
    )
    parser.add_argument(
        "--csv-log-interval",
        type=int,
        default=1000,
        help="Write progress metrics to CSV every N training steps. Use 0 to disable.",
    )

    parser.add_argument("--nyuv2-root", type=Path, default=None)
    parser.add_argument("--split-json", type=Path, default=None)
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument(
        "--augmentation",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--input-downsample-factor", type=int, default=1)
    parser.add_argument("--y1-downsample-factor", type=int, default=4)
    parser.add_argument("--y2-downsample-factor", type=int, default=4)
    parser.add_argument(
        "--y1-normalize-targets",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Standardize task-1 continuous targets as (Y - mean) / std.",
    )
    parser.add_argument(
        "--y2-normalize-targets",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Standardize task-2 continuous targets as (Y - mean) / std.",
    )
    parser.add_argument(
        "--whiten-task-inputs",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument(
        "--multimnist-source",
        choices=["torchvision", "npy"],
        default="torchvision",
        help="How to load MNIST before building MultiMNIST pairs.",
    )
    parser.add_argument("--mnist-root", type=Path, default=Path("./data"))
    parser.add_argument(
        "--mnist-train",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--mnist-download",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--multimnist-data-npy", type=Path, default=None)
    parser.add_argument("--multimnist-labels-npy", type=Path, default=None)
    parser.add_argument("--shift-range", type=int, default=4)
    parser.add_argument("--pairs-per-image", type=int, default=10)
    parser.add_argument(
        "--alignment",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-iou", type=float, default=1.0)
    parser.add_argument("--corrupt-p", type=float, default=0.0)
    parser.add_argument("--random", type=int, default=0)
    return parser.parse_args()


def load_multimnist_arrays(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    """Load MNIST arrays for MultiMNIST generation."""
    if args.multimnist_source == "npy":
        if args.multimnist_data_npy is None or args.multimnist_labels_npy is None:
            raise ValueError(
                "--multimnist-data-npy and --multimnist-labels-npy are required when "
                "--multimnist-source=npy."
            )
        data = np.load(args.multimnist_data_npy)
        labels = np.load(args.multimnist_labels_npy)
        return data, labels

    try:
        from torchvision.datasets import MNIST
    except ImportError as exc:
        raise ImportError(
            "torchvision is required for --multimnist-source=torchvision. "
            "Install torchvision or use --multimnist-source=npy."
        ) from exc

    mnist = MNIST(
        root=str(args.mnist_root),
        train=bool(args.mnist_train),
        download=bool(args.mnist_download),
    )
    return mnist.data.numpy(), mnist.targets.numpy()


def build_experiment_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Build keyword arguments for ``run_network_only_experiment``."""
    kwargs: dict[str, Any] = {
        "tasktype": args.tasktype,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "training_steps": args.training_steps,
        "learning_rate": args.learning_rate,
        "model_type": args.model_type,
        "init_scale": args.init_scale,
        "init_mode": args.init_mode,
        "init_lambda": args.init_lambda,
        "trackgradients": args.trackgradients,
        "seed": args.seed,
        "backend": args.backend,
        "prefer_gpu": args.prefer_gpu,
        "weighting_mode": args.weighting_mode,
        "gradient_surgery": args.gradient_surgery,
        "main_task": args.main_task,
        "gradnorm_alpha": args.gradnorm_alpha,
        "gradnorm_lr": args.gradnorm_lr,
        "save_weight_history": args.save_weight_history,

    }

    if args.tasktype == "multimnist":
        data, labels = load_multimnist_arrays(args)
        kwargs.update(
            {
                "data": data,
                "labels": labels,
                "train": args.mnist_train,
                "whiten_task_inputs": args.whiten_task_inputs,
                "shift_range": args.shift_range,
                "pairs_per_image": args.pairs_per_image,
                "alignment": args.alignment,
                "max_iou": args.max_iou,
                "corrupt_p": args.corrupt_p,
                "random": args.random,
            }
        )
        return kwargs

    if args.nyuv2_root is None:
        raise ValueError("--nyuv2-root is required for NYUv2 experiments.")

    kwargs.update(
        {
            "root": str(args.nyuv2_root),
            "mode": args.mode,
            "augmentation": args.augmentation,
            "input_downsample_factor": args.input_downsample_factor,
            "whiten_task_inputs": args.whiten_task_inputs,
            "y1_build_kwargs": {
                "downsample_factor": args.y1_downsample_factor,
                "normalize_targets": args.y1_normalize_targets,
            },
            "y2_build_kwargs": {
                "downsample_factor": args.y2_downsample_factor,
                "normalize_targets": args.y2_normalize_targets,
            },
            "random": args.random,
        }
    )
    if args.split_json is not None:
        kwargs["split_json"] = str(args.split_json)
    return kwargs


def build_output_stem(args: argparse.Namespace) -> str:
    """Build the shared output stem used for NPZ, JSON, and CSV files."""
    return f"{args.output_prefix}_{args.tasktype}_b{args.batch_size}_h{args.hidden_dim}_s{args.training_steps}"


def serializable_metadata(result: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Convert run metadata to a JSON-serializable dictionary."""
    metadata = dict(result.get("metadata", {}))
    metadata.update(
        {
            "tasktype": args.tasktype,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "training_steps": args.training_steps,
            "learning_rate": args.learning_rate,
            "model_type": args.model_type,
            "init_mode": args.init_mode,
            "init_lambda": args.init_lambda,
            "backend": result.get("backend"),
            "backend_resolved": result.get("backend_resolved"),
            "device": result.get("device"),
            "save_weight_history": args.save_weight_history,
            "csv_log_interval": args.csv_log_interval,
            "csv_log_path": result.get("csv_log_path"),
            "weighting_mode": result.get("weighting_mode"),
            "gradient_surgery": result.get("gradient_surgery"),
            "y1_normalize_targets": args.y1_normalize_targets,
            "y2_normalize_targets": args.y2_normalize_targets,
            "x_shape": list(result["X"].shape),
            "y1_shape": list(result["Y1"].shape),
            "y2_shape": list(result["Y2"].shape),
        }
    )
    return metadata


def build_plot_title(metadata: dict[str, Any]) -> str:
    """Build a compact plot title from stored metadata."""
    tasktype = metadata.get("tasktype", "run")
    hidden_dim = metadata.get("hidden_dim", "?")
    batch_size = metadata.get("batch_size", "?")
    weighting_mode = metadata.get("weighting_mode", "none")
    gradient_surgery = metadata.get("gradient_surgery", "none")
    return f"{tasktype} | batch={batch_size} | hidden={hidden_dim} | weights={weighting_mode} | surgery={gradient_surgery}"


def plot_saved_metrics(
    npz_path: Path,
    output_path: Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Create a metrics figure from a saved NPZ result file."""
    with np.load(npz_path) as data:
        losses1 = np.asarray(data.get("losses1", np.array([])), dtype=np.float64)
        losses2 = np.asarray(data.get("losses2", np.array([])), dtype=np.float64)
        cossim = np.asarray(data.get("cossim", np.array([])), dtype=np.float64)
        gradmag1 = np.asarray(data.get("gradmag1", np.array([])), dtype=np.float64)
        gradmag2 = np.asarray(data.get("gradmag2", np.array([])), dtype=np.float64)
        taskweight1 = np.asarray(data.get("taskweight1", np.array([])), dtype=np.float64)
        taskweight2 = np.asarray(data.get("taskweight2", np.array([])), dtype=np.float64)

    if output_path is None:
        output_path = npz_path.with_name(f"{npz_path.stem}_metrics.png")

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.ravel()

    if losses1.size > 0:
        axes[0].plot(losses1, label="Task 1")
    if losses2.size > 0:
        axes[0].plot(losses2, label="Task 2")
    axes[0].set_title("Losses")
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if cossim.size > 0:
        axes[1].plot(cossim)
    axes[1].set_title("Gradient cosine similarity")
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Cosine similarity")
    axes[1].grid(True, alpha=0.3)

    if gradmag1.size > 0:
        axes[2].plot(gradmag1, label="Task 1")
    if gradmag2.size > 0:
        axes[2].plot(gradmag2, label="Task 2")
    axes[2].set_title("Gradient magnitudes")
    axes[2].set_xlabel("Training step")
    axes[2].set_ylabel("Gradient norm")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    if taskweight1.size > 0:
        axes[3].plot(taskweight1, label="Task 1")
    if taskweight2.size > 0:
        axes[3].plot(taskweight2, label="Task 2")
    axes[3].set_title("Task weights")
    axes[3].set_xlabel("Training step")
    axes[3].set_ylabel("Weight")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    weighted_grad1 = np.array([])
    weighted_grad2 = np.array([])
    if gradmag1.size > 0 and taskweight1.size > 0:
        weighted_grad1 = taskweight1[: min(len(taskweight1), len(gradmag1))] * gradmag1[: min(len(taskweight1), len(gradmag1))]
    if gradmag2.size > 0 and taskweight2.size > 0:
        weighted_grad2 = taskweight2[: min(len(taskweight2), len(gradmag2))] * gradmag2[: min(len(taskweight2), len(gradmag2))]

    if weighted_grad1.size > 0:
        axes[4].plot(weighted_grad1, label="Task 1")
    if weighted_grad2.size > 0:
        axes[4].plot(weighted_grad2, label="Task 2")
    axes[4].set_title("Weighted gradient magnitudes")
    axes[4].set_xlabel("Training step")
    axes[4].set_ylabel("w_i * ||g_i||")
    axes[4].legend()
    axes[4].grid(True, alpha=0.3)

    axes[5].axis("off")

    if metadata is not None:
        fig.suptitle(build_plot_title(metadata))
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_result(
    result: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
    output_prefix: str,
) -> tuple[Path, Path]:
    """Save run results to NPZ and JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = build_output_stem(args)
    npz_path = output_dir / f"{stem}.npz"
    json_path = output_dir / f"{stem}.json"

    arrays: dict[str, Any] = {
        "losses1": np.asarray(result.get("losses1", []), dtype=np.float64),
        "losses2": np.asarray(result.get("losses2", []), dtype=np.float64),
        "taskweight1": np.asarray(result.get("taskweight1", []), dtype=np.float64),
        "taskweight2": np.asarray(result.get("taskweight2", []), dtype=np.float64),
    }
    if "cossim" in result:
        arrays["cossim"] = np.asarray(result["cossim"], dtype=np.float64)
    if "gradmag1" in result:
        arrays["gradmag1"] = np.asarray(result["gradmag1"], dtype=np.float64)
    if "gradmag2" in result:
        arrays["gradmag2"] = np.asarray(result["gradmag2"], dtype=np.float64)
    np.savez_compressed(npz_path, **arrays)

    metadata = serializable_metadata(result, args)
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return npz_path, json_path


def main() -> None:
    """Run the requested experiment and save outputs to disk."""
    args = parse_args()
    kwargs = build_experiment_kwargs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_log_path = args.output_dir / f"{build_output_stem(args)}_progress.csv"
    if args.csv_log_interval > 0:
        kwargs["csv_log_path"] = csv_log_path
        kwargs["csv_log_interval"] = args.csv_log_interval
    result = run_network_only_experiment(**kwargs)
    npz_path, json_path = save_result(
        result=result,
        args=args,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
    )

    plot_path = None
    if args.make_plots:
        metadata = serializable_metadata(result, args)
        plot_path = plot_saved_metrics(npz_path=npz_path, metadata=metadata)

    print(f"backend={result.get('backend')} resolved={result.get('backend_resolved')}")
    print(f"device={result.get('device', 'numpy')}")
    print(f"weighting_mode={result.get('weighting_mode')}")
    print(f"gradient_surgery={result.get('gradient_surgery', 'none')}")
    print(f"X shape={result['X'].shape}")
    print(f"Y1 shape={result['Y1'].shape}")
    print(f"Y2 shape={result['Y2'].shape}")
    if args.csv_log_interval > 0:
        print(f"saved progress csv: {csv_log_path}")
    print(f"saved npz: {npz_path}")
    print(f"saved json: {json_path}")
    if plot_path is not None:
        print(f"saved plot: {plot_path}")


if __name__ == "__main__":
    main()
