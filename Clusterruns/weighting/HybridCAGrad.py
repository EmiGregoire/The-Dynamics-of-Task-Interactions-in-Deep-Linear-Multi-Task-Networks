import numpy as np
import torch
from scipy.optimize import minimize

from LibMTL.weighting.abstract_weighting import AbsWeighting


class HybridCAGrad(AbsWeighting):
    r"""Hybrid Conflict-Averse Gradient Descent.

    Equal Weighting is applied to the uniform subset of shared parameters,
    while CAGrad is applied to the dynamic subset exposed by ``HybridHPS``.
    Task-specific parameters receive their ordinary task-loss gradients.

    Args:
        calpha (float, default=0.5): CAGrad conflict-aversion parameter.
        rescale ({0, 1, 2}, default=1): CAGrad gradient rescaling rule.
    """

    def __init__(self):
        super(HybridCAGrad, self).__init__()

    def _cagrad_weights(self, grads, calpha, rescale):
        gram = torch.matmul(grads, grads.t()).detach().cpu()
        mean_grad_norm = (gram.mean() + 1e-8).sqrt()

        start = np.ones(self.task_num, dtype=np.float64) / self.task_num
        bounds = tuple((0.0, 1.0) for _ in range(self.task_num))
        constraint = {"type": "eq", "fun": lambda weights: 1.0 - weights.sum()}
        gram_numpy = gram.numpy()
        c = float(calpha * mean_grad_norm + 1e-8)

        def objective(weights):
            weights = weights.reshape(1, -1)
            mean_weights = start.reshape(-1, 1)
            linear_term = weights.dot(gram_numpy).dot(mean_weights)
            quadratic_term = weights.dot(gram_numpy).dot(weights.T)
            return float(linear_term + c * np.sqrt(quadratic_term + 1e-8))

        result = minimize(
            objective,
            start,
            bounds=bounds,
            constraints=constraint,
        )
        if not result.success:
            raise RuntimeError(f"CAGrad optimization failed: {result.message}")

        cagrad_weights = torch.as_tensor(
            result.x,
            dtype=grads.dtype,
            device=self.device,
        )
        weighted_grad = (grads * cagrad_weights.unsqueeze(1)).sum(0)
        mixing = c / (weighted_grad.norm() + 1e-8)
        effective_weights = (
            torch.full_like(cagrad_weights, 1.0 / self.task_num)
            + mixing * cagrad_weights
        )

        if rescale == 1:
            effective_weights = effective_weights / (1.0 + calpha ** 2)
        elif rescale == 2:
            effective_weights = effective_weights / (1.0 + calpha)
        elif rescale != 0:
            raise ValueError(f"No support rescale type {rescale}")

        return effective_weights, result.x

    def _two_layer_relu_backward(self, losses, calpha, rescale):
        """Channel-wise HybridCAGrad for the lightweight NYUv2 encoder.

        The two-layer-ReLU encoder has one shared convolution whose output
        channels are partitioned by ``HybridHPS``.  CAGrad is solved using
        only the dynamic-channel gradient vectors.  Uniform channels keep the
        ordinary EW update. At fraction 0 the two-layer-ReLU endpoint uses
        normalized EW (mean task loss), so it is directly comparable with the
        normalized EW baselines used in these experiments.
        """
        partition = self.get_two_layer_relu_channel_partition()
        if partition['dynamic_channels'] == 0:
            # Exactly normalized EW for every parameter of the two-layer model.
            losses.mean().backward()
            return np.full(self.task_num, 1.0 / self.task_num)

        params = list(partition['dynamic_parameter_masks'].keys())
        masks = partition['dynamic_parameter_masks']
        flat_grads, per_parameter = [], [[] for _ in params]
        for task_idx in range(self.task_num):
            grad_list = torch.autograd.grad(
                losses[task_idx], params, retain_graph=True, allow_unused=True,
            )
            flat_grads.append(torch.cat([
                ((grad if grad is not None else param.new_zeros(param.shape)) * masks[param]).reshape(-1)
                for grad, param in zip(grad_list, params)
            ]))
            for param_idx, (grad, param) in enumerate(zip(grad_list, params)):
                per_parameter[param_idx].append(
                    grad if grad is not None else param.new_zeros(param.shape)
                )

        dynamic_grads = torch.stack(flat_grads, dim=0)
        effective_weights, reported_weights = self._cagrad_weights(
            dynamic_grads, calpha, rescale,
        )

        # This supplies each task-specific decoder with its ordinary raw task
        # gradient, exactly as standard CAGrad's per-task backward passes do.
        losses.sum().backward()

        for param, task_grads in zip(params, per_parameter):
            task_grads = torch.stack(task_grads, dim=0)
            dynamic_grad = torch.sum(
                effective_weights.view(self.task_num, *([1] * (task_grads.ndim - 1))) * task_grads,
                dim=0,
            )
            uniform_grad = task_grads.sum(dim=0)
            param.grad.detach().copy_(torch.where(masks[param], dynamic_grad, uniform_grad))

        return reported_weights

    def backward(self, losses, **kwargs):
        if (
            hasattr(self, '_is_two_layer_relu_encoder')
            and self._is_two_layer_relu_encoder()
        ):
            if self.rep_grad:
                raise ValueError(
                    'HybridCAGrad does not support representation gradients '
                    '(rep_grad=True)'
                )
            return self._two_layer_relu_backward(
                losses, kwargs['calpha'], kwargs['rescale']
            )

        required_methods = [
            "get_share_params",
            "get_dynamic_share_params",
            "get_uniform_share_params",
        ]
        if any(not hasattr(self, name) for name in required_methods):
            raise ValueError(
                "HybridCAGrad requires an architecture exposing shared, "
                "dynamic, and uniform parameter groups"
            )
        if self.rep_grad:
            raise ValueError(
                "HybridCAGrad does not support representation gradients "
                "(rep_grad=True)"
            )

        calpha = kwargs["calpha"]
        rescale = kwargs["rescale"]
        dynamic_params = list(self.get_dynamic_share_params())
        uniform_params = list(self.get_uniform_share_params())

        # fraction=0 is exactly EW and avoids running CAGrad on an empty vector.
        if len(dynamic_params) == 0:
            losses.sum().backward()
            return np.ones(self.task_num)

        uniform_grads = None
        if len(uniform_params) > 0:
            uniform_grads = self._get_grads(
                losses,
                mode="autograd",
                params=uniform_params,
            )

        # Backward mode also populates gradients for the task-specific decoders.
        dynamic_grads = self._get_grads(
            losses,
            mode="backward",
            params=dynamic_params,
        )
        effective_weights, reported_weights = self._cagrad_weights(
            dynamic_grads,
            calpha,
            rescale,
        )
        self._backward_new_grads(
            effective_weights,
            grads=dynamic_grads,
            params=dynamic_params,
        )

        if uniform_grads is not None:
            self._backward_new_grads(
                torch.ones_like(losses, device=self.device),
                grads=uniform_grads,
                params=uniform_params,
            )

        return reported_weights
