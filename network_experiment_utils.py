from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import zoom

from datasets import get_multimnist_tasks, get_nyuv2_tasks
from networks.linear_network import MTL_linearnetwork
from networks.relu_network import MTLReLUNetwork
from networks.torch_mtl_network import get_preferred_device, run_torch_mtl_experiment
from progress_logging import CsvProgressLogger
from task_overlap import build_label_matrix, whiten_inputs
from utils import MTL_get_lambda_balanced_STLex


def prepare_network_task_data(
    tasktype: str,
    batch_size: int,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Prepare dataset-backed MTL data for the notebook training loop.

    The returned arrays follow the notebook's sample-major convention:
    - ``X`` has shape ``(batch_size, input_dim)``
    - ``Y1`` has shape ``(batch_size, output_dim_1)``
    - ``Y2`` has shape ``(batch_size, output_dim_2)``

    The network can then be called exactly as in the notebook with
    ``MTLmodel.trainMTL(X.T, Y1.T, Y2.T, ...)``.
    """
    tasktype = tasktype.lower()
    if tasktype == "multimnist":
        return _prepare_multimnist_data(batch_size=batch_size, **kwargs)
    if tasktype == "nyuv2_depth_normal":
        return _prepare_nyuv2_data(
            batch_size=batch_size,
            tasks=("depth", "normal"),
            y1_task_type="depth",
            y2_task_type="normals",
            **kwargs,
        )
    if tasktype == "nyuv2_segmentation_depth":
        return _prepare_nyuv2_data(
            batch_size=batch_size,
            tasks=("segmentation", "depth"),
            y1_task_type="segmentation",
            y2_task_type="depth",
            **kwargs,
        )
    if tasktype == "nyuv2_segmentation_normal":
        return _prepare_nyuv2_data(
            batch_size=batch_size,
            tasks=("segmentation", "normal"),
            y1_task_type="segmentation",
            y2_task_type="normals",
            **kwargs,
        )
    raise ValueError(
        f"Unsupported tasktype '{tasktype}'. "
        "Use one of {'multimnist', 'nyuv2_depth_normal', "
        "'nyuv2_segmentation_depth', 'nyuv2_segmentation_normal'}."
    )


def run_network_only_experiment(
    tasktype: str,
    batch_size: int,
    hidden_dim: int,
    training_steps: int,
    learning_rate: float,
    model_type: str = "linear",
    save_weight_history: bool = False,

    init_scale: float = 1e-4,
    init_mode: str = "random",
    init_lambda: float = 0.0,
    trackgradients: bool = True,
    seed: int = 0,
    backend: str = "auto",
    prefer_gpu: bool = True,
    weighting_mode: str = "none",
    gradient_surgery: str = "none",
    main_task: int = 1,
    gradnorm_alpha: float = 1.5,
    gradnorm_lr: float = 0.025,
    csv_log_path: str | Path | None = None,
    csv_log_interval: int = 1000,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run the empirical network-only MTL experiment on dataset-backed tasks."""
    X, Y1, Y2, metadata = prepare_network_task_data(
        tasktype=tasktype,
        batch_size=batch_size,
        **kwargs,
    )

    resolved_backend = _resolve_backend(
        backend=backend,
        prefer_gpu=prefer_gpu,
        hidden_dim=hidden_dim,
    )

    progress_logger = (
        CsvProgressLogger(Path(csv_log_path), csv_log_interval)
        if csv_log_path is not None and csv_log_interval > 0
        else None
    )

    if resolved_backend == "torch":
        init_ws, init_w1, init_w2 = _make_initial_weights(
            input_dim=X.shape[1],
            hidden_dim=hidden_dim,
            out_dim_1=Y1.shape[1],
            out_dim_2=Y2.shape[1],
            init_scale=init_scale,
            init_mode=init_mode,
            init_lambda=init_lambda,
            seed=seed,
        )
        try:
            torch_result = run_torch_mtl_experiment(
                X=X,
                Y1=Y1,
                Y2=Y2,
                hidden_dim=hidden_dim,
                training_steps=training_steps,
                learning_rate=learning_rate,
                model_type=model_type,
                init_scale=init_scale,
                save_weight_history=save_weight_history,
                seed=seed,
                prefer_gpu=prefer_gpu,
                trackgradients=trackgradients,
                weighting_mode=weighting_mode,
                gradient_surgery=gradient_surgery,
                main_task=main_task,
                gradnorm_alpha=gradnorm_alpha,
                gradnorm_lr=gradnorm_lr,
                progress_callback=progress_logger,
                init_ws=init_ws,
                init_w1=init_w1,
                init_w2=init_w2,
            )
        finally:
            if progress_logger is not None:
                progress_logger.close()
        torch_result.update(
            {
                "X": X,
                "Y1": Y1,
                "Y2": Y2,
                "metadata": metadata,
                "model_type": model_type,
                "tasktype": tasktype,
                "backend_resolved": resolved_backend,
                "weighting_mode": weighting_mode,
                "gradient_surgery": gradient_surgery,
                "csv_log_path": str(csv_log_path) if csv_log_path is not None else None,
            }
        )
        return torch_result

    rng = np.random.default_rng(seed)
    init_ws, init_w1, init_w2 = _make_initial_weights(
        input_dim=X.shape[1],
        hidden_dim=hidden_dim,
        out_dim_1=Y1.shape[1],
        out_dim_2=Y2.shape[1],
        init_scale=init_scale,
        init_mode=init_mode,
        init_lambda=init_lambda,
        seed=seed,
        rng=rng,
    )

    if model_type == "linear":
        model = MTL_linearnetwork(
            X.shape[1],
            hidden_dim,
            Y1.shape[1],
            Y2.shape[1],
            init_ws=init_ws,
            init_w1=init_w1,
            init_w2=init_w2,
        )
    elif model_type == "relu":
        model = MTLReLUNetwork(
            X.shape[1],
            hidden_dim,
            Y1.shape[1],
            Y2.shape[1],
            init_ws=init_ws,
            init_w1=init_w1,
            init_w2=init_w2,
        )
    else:
        raise ValueError("model_type must be 'linear' or 'relu'.")

    try:
        outputs = model.trainMTL(
            X.T,
            Y1.T,
            Y2.T,
            training_steps,
            learning_rate,
            trackgradients=trackgradients,
            weighting_mode=weighting_mode,
            gradient_surgery=gradient_surgery,
            main_task=main_task,
            gradnorm_alpha=gradnorm_alpha,
            gradnorm_lr=gradnorm_lr,
            progress_callback=progress_logger,
            save_weight_history=save_weight_history,
        )
    finally:
        if progress_logger is not None:
            progress_logger.close()

    result = {
        "X": X,
        "Y1": Y1,
        "Y2": Y2,
        "metadata": metadata,
        "model_type": model_type,
        "tasktype": tasktype,
        "backend": "numpy",
        "backend_resolved": resolved_backend,
        "weighting_mode": weighting_mode,
        "gradient_surgery": gradient_surgery,
        "csv_log_path": str(csv_log_path) if csv_log_path is not None else None,
        "taskweight1": list(getattr(model, "taskweight1_history", [])),
        "taskweight2": list(getattr(model, "taskweight2_history", [])),
    }

    if trackgradients:
        (
            wslist,
            w1list,
            w2list,
            losses1,
            losses2,
            cossim,
            gradmag1,
            gradmag2,
        ) = outputs
        result.update(
            {
                "wslist": wslist,
                "w1list": w1list,
                "w2list": w2list,
                "losses1": losses1,
                "losses2": losses2,
                "cossim": cossim,
                "gradmag1": gradmag1,
                "gradmag2": gradmag2,
            }
        )
    else:
        wslist, w1list, w2list, losses1, losses2 = outputs
        result.update(
            {
                "wslist": wslist,
                "w1list": w1list,
                "w2list": w2list,
                "losses1": losses1,
                "losses2": losses2,
            }
        )
    return result


def _make_initial_weights(
    input_dim: int,
    hidden_dim: int,
    out_dim_1: int,
    out_dim_2: int,
    init_scale: float,
    init_mode: str,
    init_lambda: float,
    seed: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    init_mode = init_mode.lower()
    if init_mode == "random":
        if rng is None:
            rng = np.random.default_rng(seed)
        init_ws = init_scale * rng.standard_normal((hidden_dim, input_dim))
        init_w1 = init_scale * rng.standard_normal((out_dim_1, hidden_dim))
        init_w2 = init_scale * rng.standard_normal((out_dim_2, hidden_dim))
        return init_ws, init_w1, init_w2

    if init_mode == "balanced":
        state = np.random.get_state()
        np.random.seed(seed)
        try:
            init_ws, init_w1, init_w2 = MTL_get_lambda_balanced_STLex(
                init_lambda,
                input_dim,
                hidden_dim,
                out_dim_1,
                out_dim_2,
                sigma_weights=init_scale,
            )
        finally:
            np.random.set_state(state)
        return (
            np.asarray(init_ws, dtype=np.float64),
            np.asarray(init_w1, dtype=np.float64),
            np.asarray(init_w2, dtype=np.float64),
        )

    raise ValueError("init_mode must be 'random' or 'balanced'.")


def _resolve_backend(backend: str, prefer_gpu: bool, hidden_dim: int) -> str:
    backend = backend.lower()
    if backend not in {"auto", "numpy", "torch"}:
        raise ValueError("backend must be one of {'auto', 'numpy', 'torch'}.")

    if backend == "numpy":
        return "numpy"
    if backend == "torch":
        return "torch"

    if prefer_gpu and str(get_preferred_device(prefer_gpu=True)) == "cuda":
        return "torch"
    if hidden_dim >= 1000:
        return "torch"
    return "numpy"


def _prepare_multimnist_data(
    batch_size: int,
    data: np.ndarray,
    labels: np.ndarray,
    train: bool = True,
    whiten_task_inputs: bool = False,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    X, Y1, Y2 = get_multimnist_tasks(
        data=data,
        labels=labels,
        batch_size=batch_size,
        train=train,
        whiten_inputs=whiten_task_inputs,
        **{k: v for k, v in kwargs.items() if k in {
            "shift_range",
            "pairs_per_image",
            "alignment",
            "max_iou",
            "corrupt_p",
            "random",
        }},
    )
    return X, Y1.T, Y2.T, {
        "dataset": "multimnist",
        "input_dim": X.shape[1],
        "out_dim_1": Y1.shape[0],
        "out_dim_2": Y2.shape[0],
    }


def _prepare_nyuv2_data(
    batch_size: int,
    tasks: tuple[str, str],
    y1_task_type: str,
    y2_task_type: str,
    root: str,
    mode: str = "train",
    augmentation: bool = False,
    input_downsample_factor: int | tuple[int, int] | None = None,
    whiten_task_inputs: bool = False,
    y1_build_kwargs: dict[str, Any] | None = None,
    y2_build_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    X_raw, Y1_raw, Y2_raw = get_nyuv2_tasks(
        root=root,
        batch_size=batch_size,
        mode=mode,
        augmentation=augmentation,
        tasks=tasks,
        mtl_format=False,
        split_json=kwargs.get("split_json"),
        random=kwargs.get("random"),
    )

    X = _flatten_inputs(
        X_raw,
        downsample_factor=input_downsample_factor,
        whiten_task_inputs=whiten_task_inputs,
    )
    Y1 = build_label_matrix(
        Y1_raw,
        y1_task_type,
        **(y1_build_kwargs or {}),
    )
    Y2 = build_label_matrix(
        Y2_raw,
        y2_task_type,
        **(y2_build_kwargs or {}),
    )
    return X, Y1, Y2, {
        "dataset": "nyuv2",
        "tasks": tasks,
        "input_dim": X.shape[1],
        "out_dim_1": Y1.shape[1],
        "out_dim_2": Y2.shape[1],
    }


def _flatten_inputs(
    X: np.ndarray,
    downsample_factor: int | tuple[int, int] | None = None,
    whiten_task_inputs: bool = False,
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 4 and downsample_factor is not None:
        if np.isscalar(downsample_factor):
            factor_h = factor_w = float(downsample_factor)
        else:
            factor_h, factor_w = float(downsample_factor[0]), float(downsample_factor[1])
        X = zoom(X, zoom=(1.0, 1.0, 1.0 / factor_h, 1.0 / factor_w), order=1)

    X = X.reshape(X.shape[0], -1)
    if whiten_task_inputs:
        X, _ = whiten_inputs(X)
    return X
