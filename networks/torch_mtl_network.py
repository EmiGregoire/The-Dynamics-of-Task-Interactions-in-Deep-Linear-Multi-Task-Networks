from __future__ import annotations

from typing import Any
from unittest import result

import numpy as np
import torch
import torch.nn as nn


def get_preferred_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _normalize_task_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.clip(weights, 1e-8, None)
    return weights / np.sum(weights)


def _pcgrad_two_task_torch(
    g1: torch.Tensor,
    g2: torch.Tensor,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project conflicting shared gradients using the two-task PCGrad rule."""
    g1_orig = g1.detach().clone()
    g2_orig = g2.detach().clone()

    v1 = g1_orig.reshape(-1)
    v2 = g2_orig.reshape(-1)
    dot12 = torch.dot(v1, v2)
    if float(dot12.detach().cpu()) >= 0.0:
        return g1_orig, g2_orig

    g1_proj = g1_orig - (dot12 / (torch.dot(v2, v2) + eps)) * g2_orig
    g2_proj = g2_orig - (dot12 / (torch.dot(v1, v1) + eps)) * g1_orig
    return g1_proj, g2_proj


def _compute_next_task_weights(
    weighting_mode: str,
    current_weights: np.ndarray,
    gradmag1: float,
    gradmag2: float,
    cos_sim: float,
    loss1: float,
    loss2: float,
    state: dict[str, Any],
    main_task: int = 1,
    gradnorm_alpha: float = 1.5,
    gradnorm_lr: float = 0.025,
) -> tuple[np.ndarray, dict[str, Any]]:
    eps = 1e-12
    mode = weighting_mode.lower()

    if mode == "none":
        return np.array([0.5, 0.5], dtype=np.float64), state

    if mode == "cossim":
        aux_weight = max(0.0, float(cos_sim))
        if main_task == 1:
            return _normalize_task_weights(np.array([1.0, aux_weight], dtype=np.float64)), state
        if main_task == 2:
            return _normalize_task_weights(np.array([aux_weight, 1.0], dtype=np.float64)), state
        raise ValueError("main_task must be 1 or 2 for cossim weighting.")

    if mode == "gradnorm":
        losses = np.array([max(loss1, eps), max(loss2, eps)], dtype=np.float64)
        if state.get("initial_losses") is None:
            state["initial_losses"] = losses.copy()
            current_weights = np.array([0.5, 0.5], dtype=np.float64)

        base_grad_norms = np.array([gradmag1, gradmag2], dtype=np.float64)
        grad_norms = current_weights * base_grad_norms
        grad_norm_mean = np.mean(grad_norms)

        loss_ratios = losses / state["initial_losses"]
        inverse_train_rates = loss_ratios / np.mean(loss_ratios)
        targets = grad_norm_mean * np.power(inverse_train_rates, gradnorm_alpha)

        grad_w = np.sign(grad_norms - targets) * base_grad_norms
        next_weights = current_weights - gradnorm_lr * grad_w
        return _normalize_task_weights(next_weights), state

    raise ValueError("weighting_mode must be one of {'none', 'gradnorm', 'cossim'}.")


class TorchMTLLinearNetwork(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim_1: int,
        out_dim_2: int,
        init_scale: float = 1e-3,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)

        self.WS = nn.Parameter(init_scale * torch.randn(hidden_dim, in_dim))
        self.W1 = nn.Parameter(init_scale * torch.randn(out_dim_1, hidden_dim))
        self.W2 = nn.Parameter(init_scale * torch.randn(out_dim_2, hidden_dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = x @ self.WS.T
        t1 = z @ self.W1.T
        t2 = z @ self.W2.T
        return t1, t2


class TorchMTLReLUNetwork(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim_1: int,
        out_dim_2: int,
        init_scale: float = 1e-3,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)

        self.WS = nn.Parameter(init_scale * torch.randn(hidden_dim, in_dim))
        self.W1 = nn.Parameter(init_scale * torch.randn(out_dim_1, hidden_dim))
        self.W2 = nn.Parameter(init_scale * torch.randn(out_dim_2, hidden_dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_pre = x @ self.WS.T
        z = torch.relu(z_pre)
        t1 = z @ self.W1.T
        t2 = z @ self.W2.T
        return t1, t2


def run_torch_mtl_experiment(
    X: np.ndarray,
    Y1: np.ndarray,
    Y2: np.ndarray,
    hidden_dim: int,
    training_steps: int,
    learning_rate: float,
    model_type: str = "linear",
    init_scale: float = 1e-3,
    seed: int = 0,
    prefer_gpu: bool = True,
    trackgradients: bool = True,
    weighting_mode: str = "none",
    gradient_surgery: str = "none",
    main_task: int = 1,
    gradnorm_alpha: float = 1.5,
    gradnorm_lr: float = 0.025,
    save_weight_history: bool = False,
    progress_callback=None,

) -> dict[str, Any]:
    device = get_preferred_device(prefer_gpu=prefer_gpu)

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    Y1_t = torch.tensor(Y1, dtype=torch.float32, device=device)
    Y2_t = torch.tensor(Y2, dtype=torch.float32, device=device)

    if model_type == "linear":
        model = TorchMTLLinearNetwork(
            in_dim=X.shape[1],
            hidden_dim=hidden_dim,
            out_dim_1=Y1.shape[1],
            out_dim_2=Y2.shape[1],
            init_scale=init_scale,
            seed=seed,
        ).to(device)
    elif model_type == "relu":
        model = TorchMTLReLUNetwork(
            in_dim=X.shape[1],
            hidden_dim=hidden_dim,
            out_dim_1=Y1.shape[1],
            out_dim_2=Y2.shape[1],
            init_scale=init_scale,
            seed=seed,
        ).to(device)
    else:
        raise ValueError("model_type must be 'linear' or 'relu'.")

    losses1: list[float] = []
    losses2: list[float] = []
    cossim: list[float] = []
    gradmag1: list[float] = []
    gradmag2: list[float] = []
    taskweight1: list[float] = []
    taskweight2: list[float] = []
    wslist: list[np.ndarray] = []
    w1list: list[np.ndarray] = []
    w2list: list[np.ndarray] = []
    current_weights = np.array([0.5, 0.5], dtype=np.float64)
    weight_state: dict[str, Any] = {"initial_losses": None, "task_weights": current_weights.copy()}

    for step in range(training_steps):
        model.zero_grad()
        pred1, pred2 = model(X_t)

        loss1 = torch.mean((pred1 - Y1_t) ** 2)
        loss2 = torch.mean((pred2 - Y2_t) ** 2)

        losses1.append(float(loss1.detach().cpu()))
        losses2.append(float(loss2.detach().cpu()))

        if save_weight_history:
            wslist.append(model.WS.detach().cpu().numpy().copy())
            w1list.append(model.W1.detach().cpu().numpy().copy())
            w2list.append(model.W2.detach().cpu().numpy().copy())

        g1 = torch.autograd.grad(loss1, model.WS, retain_graph=True)[0]
        g2 = torch.autograd.grad(loss2, model.WS, retain_graph=True)[0]
        v1 = g1.reshape(-1)
        v2 = g2.reshape(-1)
        cos = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2) + 1e-12)
        g1_norm = float(torch.norm(v1).detach().cpu())
        g2_norm = float(torch.norm(v2).detach().cpu())
        cos_value = float(cos.detach().cpu())

        current_weights = np.asarray(weight_state["task_weights"], dtype=np.float64).copy()
        taskweight1.append(float(current_weights[0]))
        taskweight2.append(float(current_weights[1]))

        shared_g1 = current_weights[0] * g1.detach().clone()
        shared_g2 = current_weights[1] * g2.detach().clone()
        if gradient_surgery.lower() == "pcgrad":
            shared_g1, shared_g2 = _pcgrad_two_task_torch(shared_g1, shared_g2)

        total_loss = current_weights[0] * loss1 + current_weights[1] * loss2

        next_weights, weight_state = _compute_next_task_weights(
            weighting_mode=weighting_mode,
            current_weights=current_weights,
            gradmag1=g1_norm,
            gradmag2=g2_norm,
            cos_sim=cos_value,
            loss1=float(loss1.detach().cpu()),
            loss2=float(loss2.detach().cpu()),
            state=weight_state,
            main_task=main_task,
            gradnorm_alpha=gradnorm_alpha,
            gradnorm_lr=gradnorm_lr,
        )
        weight_state["task_weights"] = next_weights.copy()

        if trackgradients:
            cossim.append(cos_value)
            gradmag1.append(g1_norm)
            gradmag2.append(g2_norm)

        if progress_callback is not None:
            progress_callback(
                {
                    "step": step + 1,
                    "total_steps": training_steps,
                    "loss1": float(loss1.detach().cpu()),
                    "loss2": float(loss2.detach().cpu()),
                    "taskweight1": float(current_weights[0]),
                    "taskweight2": float(current_weights[1]),
                    "cossim": cos_value,
                    "gradmag1": g1_norm,
                    "gradmag2": g2_norm,
                }
            )

        total_loss.backward()
        model.WS.grad = (shared_g1 + shared_g2).clone()
        with torch.no_grad():
            model.WS -= learning_rate * model.WS.grad
            model.W1 -= learning_rate * model.W1.grad
            model.W2 -= learning_rate * model.W2.grad

    result = {
    "backend": "torch",
    "device": str(device),
    "model": model,
    "losses1": losses1,
    "losses2": losses2,
    "cossim": cossim,
    "gradmag1": gradmag1,
    "gradmag2": gradmag2,
    "taskweight1": taskweight1,
    "taskweight2": taskweight2,
    }
    
    if save_weight_history:
        result.update(
        {
            "wslist": wslist,
            "w1list": w1list,
            "w2list": w2list,
        }
    )
    return result
