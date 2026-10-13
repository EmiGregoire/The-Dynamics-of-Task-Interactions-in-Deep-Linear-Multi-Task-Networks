from __future__ import annotations

import itertools
import math
import warnings
from typing import Any

import numpy as np
from scipy.ndimage import zoom
from scipy.sparse.linalg import LinearOperator, eigsh
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection


EXPLICIT_M_MAX_ENTRIES = 8_000_000
KERNEL_WARN_GB = 1.0


def whiten_inputs(X: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, dict[str, Any]]:
    """Center and ZCA-whiten inputs.

    This module uses the convention ``X.shape == (N, d)`` where ``N`` is the
    number of paired samples and ``d`` is the input dimension.

    The whitening is computed with an SVD of the centered data matrix. The
    returned whitened inputs keep shape ``(N, d)`` and live in the same ambient
    input space, while directions outside the sample span naturally have zero
    variance.

    Parameters
    ----------
    X:
        Input array of shape ``(N, d)``.
    eps:
        Numerical stabilizer added to singular values.

    Returns
    -------
    Xw, metadata:
        ``Xw`` is the whitened data of shape ``(N, d)``.
        ``metadata`` contains the mean, singular values, right singular vectors,
        and the epsilon used in the transform.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Expected X to be 2D with shape (N, d), got {X.shape}.")
    if X.shape[0] < 2:
        raise ValueError("Need at least two samples to whiten inputs.")

    mean = X.mean(axis=0, keepdims=True)
    X_centered = X - mean

    U, singular_values, Vt = np.linalg.svd(X_centered, full_matrices=False)
    scale = np.sqrt(max(X.shape[0] - 1, 1))
    scaled = singular_values / (singular_values + eps)
    Xw = (U * (scale * scaled)) @ Vt

    metadata = {
        "mean": mean,
        "singular_values": singular_values,
        "right_singular_vectors": Vt,
        "eps": eps,
        "method": "zca_svd",
    }
    return Xw, metadata


def build_label_matrix(task_data: Any, task_type: str, **kwargs: Any) -> np.ndarray:
    """Build a 2D sample matrix ``Y`` with shape ``(N, m)`` for a dense task.

    Supported task types are ``"depth"``, ``"surface_normals"`` / ``"normals"``,
    and ``"segmentation"``.

    Common keyword arguments
    ------------------------
    downsample_factor:
        Optional scalar or tuple applied to the spatial dimensions.
    target_shape:
        Optional output spatial size ``(H, W)``. Overrides ``downsample_factor``.
    normalize_targets:
        If ``True``, standardize the dense target matrix as
        ``(Y - Y.mean()) / Y.std()`` using a single global mean and standard
        deviation across the full batch-feature matrix. Supported for
        continuous tasks such as ``depth`` and ``normals``.
    compress:
        Optional ``None``, ``"pca"``, or ``"random_projection"``.
    rank:
        Target rank used by the compression step.
    random_state:
        Random seed used by random projection.

    Segmentation-specific keyword arguments
    ---------------------------------------
    segmentation_mode:
        ``"one_hot"`` (baseline), ``"histogram"``, or ``"balanced_histogram"``.
    num_classes:
        Number of semantic classes. Inferred from the batch if omitted.
    ignore_index:
        Label value to ignore for segmentation encodings.
    """
    task_type = task_type.lower()
    if task_type in {"surface_normals", "normal"}:
        task_type = "normals"

    array = np.asarray(task_data)
    if array.shape[0] < 1:
        raise ValueError("Task data must contain at least one sample.")

    downsampled = _downsample_task_array(
        array,
        task_type=task_type,
        downsample_factor=kwargs.get("downsample_factor"),
        target_shape=kwargs.get("target_shape"),
    )

    if task_type == "segmentation":
        if kwargs.get("normalize_targets", False):
            raise ValueError(
                "normalize_targets is only supported for continuous tasks "
                "('depth' or 'normals')."
            )
        segmentation_mode = kwargs.get("segmentation_mode", "one_hot")
        num_classes = kwargs.get("num_classes")
        ignore_index = kwargs.get("ignore_index")
        Y = _segmentation_to_matrix(
            downsampled,
            mode=segmentation_mode,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )
    elif task_type in {"depth", "normals"}:
        if downsampled.ndim not in {3, 4}:
            raise ValueError(
                f"Expected {task_type} data to be 3D or 4D, got shape {downsampled.shape}."
            )
        Y = downsampled.reshape(downsampled.shape[0], -1).astype(np.float64, copy=False)
        if kwargs.get("normalize_targets", False):
            mean = float(np.mean(Y))
            std = float(np.std(Y))
            if std <= 1e-12:
                raise ValueError(
                    f"Cannot normalize {task_type} targets with near-zero std ({std})."
                )
            Y = (Y - mean) / std
    else:
        raise ValueError(
            f"Unsupported task_type '{task_type}'. "
            "Use one of {'depth', 'normals', 'segmentation'}."
        )

    return _compress_features(
        Y,
        method=kwargs.get("compress"),
        rank=kwargs.get("rank"),
        random_state=kwargs.get("random_state"),
    )


def compute_label_kernel(
    Y: np.ndarray, kernel: str = "linear", **kwargs: Any
) -> np.ndarray:
    """Compute a sample kernel ``K`` from a label matrix ``Y``.

    Parameters
    ----------
    Y:
        Label matrix of shape ``(N, m)``.
    kernel:
        ``"linear"``, ``"normalized_linear"``, or ``"rbf"``.

    Other Parameters
    ----------------
    center:
        If ``True``, center the resulting kernel in sample space.
    gamma:
        Optional RBF bandwidth. Defaults to ``1 / m``.
    block_size:
        Optional block size for the linear kernel computation.
    """
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim != 2:
        raise ValueError(f"Expected Y to be 2D with shape (N, m), got {Y.shape}.")

    n_samples = Y.shape[0]
    approx_gb = (n_samples * n_samples * 8) / (1024**3)
    if approx_gb > KERNEL_WARN_GB:
        warnings.warn(
            f"Kernel matrix K will occupy about {approx_gb:.2f} GB in float64.",
            RuntimeWarning,
        )

    kernel = kernel.lower()
    if kernel == "linear":
        K = _linear_kernel(Y, block_size=kwargs.get("block_size"))
    elif kernel == "normalized_linear":
        norms = np.linalg.norm(Y, axis=1, keepdims=True)
        Y_normalized = Y / np.clip(norms, 1e-12, None)
        K = _linear_kernel(Y_normalized, block_size=kwargs.get("block_size"))
    elif kernel == "rbf":
        gamma = kwargs.get("gamma")
        if gamma is None:
            gamma = 1.0 / max(Y.shape[1], 1)
        sq_norms = np.sum(Y * Y, axis=1, keepdims=True)
        sq_dists = sq_norms + sq_norms.T - 2.0 * (Y @ Y.T)
        sq_dists = np.maximum(sq_dists, 0.0)
        K = np.exp(-gamma * sq_dists)
    else:
        raise ValueError(
            f"Unsupported kernel '{kernel}'. "
            "Use one of {'linear', 'normalized_linear', 'rbf'}."
        )

    if kwargs.get("center", False):
        K = _center_kernel(K)

    K = 0.5 * (K + K.T)
    return K


def matvec_M(v: np.ndarray, Xw: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Apply the task matrix ``M = Xw.T @ K @ Xw`` to a vector or matrix.

    Shapes
    ------
    - ``Xw`` has shape ``(N, d)``
    - ``K`` has shape ``(N, N)``
    - ``v`` has shape ``(d,)`` or ``(d, r)``

    The multiplication is evaluated as
    ``Xw.T @ (K @ (Xw @ v))`` to avoid explicitly forming ``M`` when ``d`` is
    large.
    """
    Xw = np.asarray(Xw, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    if Xw.ndim != 2:
        raise ValueError(f"Expected Xw to have shape (N, d), got {Xw.shape}.")
    if K.shape != (Xw.shape[0], Xw.shape[0]):
        raise ValueError(
            f"K must have shape ({Xw.shape[0]}, {Xw.shape[0]}), got {K.shape}."
        )

    if v.ndim == 1:
        return Xw.T @ (K @ (Xw @ v))
    if v.ndim == 2:
        return Xw.T @ (K @ (Xw @ v))
    raise ValueError(f"Expected v to be 1D or 2D, got {v.shape}.")


def top_task_subspace(
    Xw: np.ndarray, K: np.ndarray, rank: int
) -> tuple[np.ndarray, np.ndarray]:
    """Recover the leading task subspace in input space.

    The eigenspace is defined by the top eigenvectors of
    ``M = Xw.T @ K @ Xw``. This function automatically avoids forming ``M`` when
    the input dimension is large by using ``scipy.sparse.linalg.eigsh`` on a
    ``LinearOperator`` built from :func:`matvec_M`.

    Returns
    -------
    V, eigenvalues:
        ``V`` has shape ``(d, rank_eff)`` and contains orthonormal principal
        directions in input space. ``eigenvalues`` are sorted in descending
        order.
    """
    Xw = np.asarray(Xw, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    if Xw.ndim != 2:
        raise ValueError(f"Expected Xw to have shape (N, d), got {Xw.shape}.")
    if K.shape != (Xw.shape[0], Xw.shape[0]):
        raise ValueError(
            f"K must have shape ({Xw.shape[0]}, {Xw.shape[0]}), got {K.shape}."
        )
    if rank < 1:
        raise ValueError(f"rank must be positive, got {rank}.")

    K = 0.5 * (K + K.T)
    d = Xw.shape[1]
    rank_eff = min(rank, d, max(Xw.shape[0] - 1, 1))

    if d * d <= EXPLICIT_M_MAX_ENTRIES or rank_eff >= d - 1:
        M = Xw.T @ (K @ Xw)
        M = 0.5 * (M + M.T)
        eigenvalues, eigenvectors = np.linalg.eigh(M)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order][:rank_eff]
        eigenvectors = eigenvectors[:, order][:, :rank_eff]
    else:
        operator = LinearOperator(
            shape=(d, d),
            matvec=lambda v: matvec_M(v, Xw, K),
            matmat=lambda V: matvec_M(V, Xw, K),
            dtype=np.float64,
        )
        eigenvalues, eigenvectors = eigsh(operator, k=rank_eff, which="LA")
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

    positive = eigenvalues > 1e-10
    eigenvalues = eigenvalues[positive]
    eigenvectors = eigenvectors[:, positive]

    if eigenvectors.size == 0:
        raise ValueError("No positive eigenvalues were found for the task matrix.")

    eigenvectors /= np.linalg.norm(eigenvectors, axis=0, keepdims=True).clip(min=1e-12)
    return eigenvectors, eigenvalues


def task_overlap(V1: np.ndarray, V2: np.ndarray) -> dict[str, np.ndarray]:
    """Compute overlap statistics between two task subspaces."""
    V1 = np.asarray(V1, dtype=np.float64)
    V2 = np.asarray(V2, dtype=np.float64)
    if V1.ndim != 2 or V2.ndim != 2:
        raise ValueError("V1 and V2 must both be 2D arrays of subspace vectors.")
    if V1.shape[0] != V2.shape[0]:
        raise ValueError(
            f"Subspaces must live in the same input space, got {V1.shape} and {V2.shape}."
        )

    O = V1.T @ V2
    singular_values = np.linalg.svd(O, compute_uv=False)
    singular_values = np.clip(singular_values, 0.0, 1.0)
    principal_angles_deg = np.degrees(np.arccos(singular_values))
    return {
        "overlap_matrix": O,
        "singular_values": singular_values,
        "principal_angles_deg": principal_angles_deg,
    }


def task_magnitude_metrics(
    Xw: np.ndarray, K: np.ndarray, top_rank: int = 8
) -> dict[str, float | np.ndarray]:
    """Compute task-magnitude summaries from the low-rank sample form.

    With the module convention ``Xw.shape == (N, d)`` and ``Y.shape == (N, m)``,
    the empirical task matrix is

    ``Sigma = Xw.T @ Y / N``.

    Writing ``K = Y @ Y.T``, we have

    ``Sigma @ Sigma.T = (1 / N^2) * Xw.T @ K @ Xw``.

    This function reports magnitude summaries of ``Sigma`` without explicitly
    forming ``Sigma``:

    - Frobenius norm
    - spectral norm
    - trace of ``Sigma Sigma.T``
    - top eigenvalues of ``Sigma Sigma.T``
    - stable rank = ||Sigma||_F^2 / ||Sigma||_2^2
    """
    Xw = np.asarray(Xw, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    if Xw.ndim != 2:
        raise ValueError(f"Expected Xw to have shape (N, d), got {Xw.shape}.")
    if K.shape != (Xw.shape[0], Xw.shape[0]):
        raise ValueError(
            f"K must have shape ({Xw.shape[0]}, {Xw.shape[0]}), got {K.shape}."
        )

    n_samples = Xw.shape[0]
    Gx = Xw @ Xw.T
    trace_M = float(np.sum(K * Gx))
    trace_sigma = max(trace_M / (n_samples**2), 0.0)
    fro_sq = trace_sigma
    fro = math.sqrt(fro_sq)

    _, evals = top_task_subspace(Xw, K, rank=max(1, top_rank))
    sigma_sq_eigs = evals / (n_samples**2)
    sigma_sq_eigs = np.maximum(sigma_sq_eigs, 0.0)
    sigma_eigs = np.sqrt(sigma_sq_eigs)
    spectral = float(sigma_eigs[0])
    stable_rank = fro_sq / max(spectral**2, 1e-12)

    return {
        "frobenius_norm": fro,
        "frobenius_norm_sq": fro_sq,
        "spectral_norm": spectral,
        "trace_sigma_sigma_t": trace_sigma,
        "stable_rank": stable_rank,
        "top_singular_values": sigma_eigs,
        "top_eigenvalues_sigma_sigma_t": sigma_sq_eigs,
    }


def magnitude_imbalance(
    task_metrics: dict[str, dict[str, float | np.ndarray]],
    metric: str = "frobenius_norm",
) -> dict[str, Any]:
    """Summarize magnitude imbalance across tasks.

    Parameters
    ----------
    task_metrics:
        Mapping from task name to the output of :func:`task_magnitude_metrics`.
    metric:
        Which scalar metric to compare across tasks. Typical choices are
        ``"frobenius_norm"``, ``"spectral_norm"``, or ``"stable_rank"``.

    Returns
    -------
    dict:
        Contains per-task metric values and aggregate imbalance summaries such as
        max/min ratio, coefficient of variation, and normalized gap.
    """
    if not task_metrics:
        raise ValueError("task_metrics must contain at least one task.")

    values = {
        task: float(metrics[metric])  # type: ignore[arg-type]
        for task, metrics in task_metrics.items()
    }
    arr = np.asarray(list(values.values()), dtype=np.float64)

    max_value = float(np.max(arr))
    min_value = float(np.min(arr))
    mean_value = float(np.mean(arr))
    std_value = float(np.std(arr))

    return {
        "metric": metric,
        "per_task": values,
        "max_value": max_value,
        "min_value": min_value,
        "mean_value": mean_value,
        "std_value": std_value,
        "max_min_ratio": max_value / max(min_value, 1e-12),
        "normalized_gap": (max_value - min_value) / max(mean_value, 1e-12),
        "coefficient_of_variation": std_value / max(mean_value, 1e-12),
        "log10_max_min_ratio": math.log10(max_value / max(min_value, 1e-12)),
    }


def pairwise_task_overlap(
    task_subspaces: dict[str, np.ndarray]
) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    """Compute pairwise overlap statistics across a dictionary of task subspaces."""
    overlaps: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for task_a, task_b in itertools.combinations(sorted(task_subspaces), 2):
        overlaps[(task_a, task_b)] = task_overlap(
            task_subspaces[task_a], task_subspaces[task_b]
        )
    return overlaps


def _downsample_task_array(
    array: np.ndarray,
    task_type: str,
    downsample_factor: Any = None,
    target_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Downsample dense task arrays while preserving task semantics."""
    if target_shape is None and downsample_factor is None:
        return np.asarray(array)

    array = np.asarray(array)
    spatial_shape = _extract_spatial_shape(array, task_type)
    if target_shape is None:
        if np.isscalar(downsample_factor):
            target_shape = tuple(
                max(1, int(math.ceil(dim / float(downsample_factor))))
                for dim in spatial_shape
            )
        else:
            factor_h, factor_w = downsample_factor
            target_shape = (
                max(1, int(math.ceil(spatial_shape[0] / float(factor_h)))),
                max(1, int(math.ceil(spatial_shape[1] / float(factor_w)))),
            )

    if array.ndim == 3:
        zoom_factors = (1.0, target_shape[0] / spatial_shape[0], target_shape[1] / spatial_shape[1])
    elif array.ndim == 4:
        if task_type == "normals":
            zoom_factors = (
                1.0,
                1.0,
                target_shape[0] / spatial_shape[0],
                target_shape[1] / spatial_shape[1],
            )
        else:
            zoom_factors = (
                1.0,
                target_shape[0] / spatial_shape[0],
                target_shape[1] / spatial_shape[1],
                1.0,
            )
    else:
        raise ValueError(f"Unsupported task array shape for downsampling: {array.shape}.")

    order = 0 if task_type == "segmentation" else 1
    return zoom(array, zoom=zoom_factors, order=order)


def _extract_spatial_shape(array: np.ndarray, task_type: str) -> tuple[int, int]:
    if array.ndim == 3:
        return int(array.shape[-2]), int(array.shape[-1])
    if array.ndim == 4:
        if task_type == "normals":
            return int(array.shape[-2]), int(array.shape[-1])
        return int(array.shape[1]), int(array.shape[2])
    raise ValueError(f"Unsupported task array shape {array.shape}.")


def _segmentation_to_matrix(
    masks: np.ndarray,
    mode: str = "one_hot",
    num_classes: int | None = None,
    ignore_index: int | None = None,
) -> np.ndarray:
    """Convert segmentation masks to a sample-feature matrix."""
    masks = np.asarray(masks)
    if masks.ndim != 3:
        raise ValueError(f"Expected segmentation masks with shape (N, H, W), got {masks.shape}.")

    mode = mode.lower()
    valid_masks = masks[masks >= 0]
    if ignore_index is not None:
        valid_masks = valid_masks[valid_masks != ignore_index]

    if num_classes is None:
        if valid_masks.size == 0:
            raise ValueError("Cannot infer num_classes from an empty valid mask set.")
        num_classes = int(valid_masks.max()) + 1

    if mode == "one_hot":
        flat = masks.reshape(masks.shape[0], -1)
        out = np.zeros((flat.shape[0], flat.shape[1] * num_classes), dtype=np.float64)
        valid = flat >= 0
        if ignore_index is not None:
            valid &= flat != ignore_index
        rows, cols = np.nonzero(valid)
        class_ids = flat[rows, cols].astype(int)
        out[rows, cols * num_classes + class_ids] = 1.0
        return out

    hist = np.zeros((masks.shape[0], num_classes), dtype=np.float64)
    for idx, mask in enumerate(masks):
        flat = mask.reshape(-1)
        flat = flat[flat >= 0]
        if ignore_index is not None:
            flat = flat[flat != ignore_index]
        if flat.size == 0:
            continue
        bincount = np.bincount(flat.astype(int), minlength=num_classes)
        hist[idx] = bincount[:num_classes]

    if mode == "histogram":
        hist /= hist.sum(axis=1, keepdims=True).clip(min=1e-12)
        return hist
    if mode == "balanced_histogram":
        global_hist = hist.sum(axis=0, keepdims=True)
        weights = 1.0 / np.sqrt(global_hist + 1e-12)
        balanced = hist * weights
        balanced /= balanced.sum(axis=1, keepdims=True).clip(min=1e-12)
        return balanced

    raise ValueError(
        f"Unsupported segmentation_mode '{mode}'. "
        "Use one of {'one_hot', 'histogram', 'balanced_histogram'}."
    )


def _compress_features(
    Y: np.ndarray,
    method: str | None = None,
    rank: int | None = None,
    random_state: int | None = None,
) -> np.ndarray:
    """Optionally compress a label matrix before kernel construction."""
    Y = np.asarray(Y, dtype=np.float64)
    if method is None or rank is None:
        return Y

    rank = min(rank, Y.shape[0], Y.shape[1])
    if rank < 1:
        raise ValueError(f"rank must be positive when compressing labels, got {rank}.")

    method = method.lower()
    if method == "pca":
        return PCA(n_components=rank, random_state=random_state).fit_transform(Y)
    if method == "random_projection":
        projector = GaussianRandomProjection(
            n_components=rank, random_state=random_state
        )
        return projector.fit_transform(Y)
    raise ValueError(
        f"Unsupported compress method '{method}'. "
        "Use one of {None, 'pca', 'random_projection'}."
    )


def _linear_kernel(Y: np.ndarray, block_size: int | None = None) -> np.ndarray:
    """Compute ``Y @ Y.T`` with optional blocking to reduce peak memory."""
    if block_size is None or block_size >= Y.shape[0]:
        K = Y @ Y.T
        return 0.5 * (K + K.T)

    n_samples = Y.shape[0]
    K = np.zeros((n_samples, n_samples), dtype=np.float64)
    for start in range(0, n_samples, block_size):
        stop = min(start + block_size, n_samples)
        K[start:stop] = Y[start:stop] @ Y.T
    return 0.5 * (K + K.T)


def _center_kernel(K: np.ndarray) -> np.ndarray:
    n_samples = K.shape[0]
    H = np.eye(n_samples) - np.ones((n_samples, n_samples)) / n_samples
    centered = H @ K @ H
    return 0.5 * (centered + centered.T)


def _synthetic_demo() -> dict[str, Any]:
    """Run a small synthetic example showing the full overlap pipeline."""
    rng = np.random.default_rng(0)
    n_samples = 48
    input_dim = 64
    output_dim = 128
    subspace_rank = 6

    X = rng.standard_normal((n_samples, input_dim))
    Xw, _ = whiten_inputs(X)

    teacher_a = rng.standard_normal((input_dim, output_dim))
    teacher_b = 0.7 * teacher_a + 0.3 * rng.standard_normal((input_dim, output_dim))

    Y_a = Xw @ teacher_a + 0.05 * rng.standard_normal((n_samples, output_dim))
    Y_b = Xw @ teacher_b + 0.05 * rng.standard_normal((n_samples, output_dim))

    K_a = compute_label_kernel(Y_a, kernel="linear", center=True)
    K_b = compute_label_kernel(Y_b, kernel="linear", center=True)

    V_a, evals_a = top_task_subspace(Xw, K_a, rank=subspace_rank)
    V_b, evals_b = top_task_subspace(Xw, K_b, rank=subspace_rank)
    overlap = task_overlap(V_a, V_b)
    imbalance = magnitude_imbalance(
        {
            "task_a": task_magnitude_metrics(Xw, K_a, top_rank=subspace_rank),
            "task_b": task_magnitude_metrics(Xw, K_b, top_rank=subspace_rank),
        }
    )

    return {
        "evals_a": evals_a,
        "evals_b": evals_b,
        "overlap": overlap,
        "imbalance": imbalance,
    }


if __name__ == "__main__":
    demo = _synthetic_demo()
    print("Top eigenvalues for task A:", np.round(demo["evals_a"], 4))
    print("Top eigenvalues for task B:", np.round(demo["evals_b"], 4))
    print("Overlap singular values:", np.round(demo["overlap"]["singular_values"], 4))
    print("Magnitude imbalance:", demo["imbalance"])
    print(
        "Principal angles (deg):",
        np.round(demo["overlap"]["principal_angles_deg"], 2),
    )
