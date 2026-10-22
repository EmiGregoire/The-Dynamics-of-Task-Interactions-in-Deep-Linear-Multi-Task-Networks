"""Channel-wise HybridGradNorm for the two-layer NYUv2 encoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from LibMTL.weighting.abstract_weighting import AbsWeighting


class HybridGradNorm(AbsWeighting):
    r"""Apply GradNorm only to the dynamic shared-channel subset.

    ``HybridHPS`` splits the output filters of the two-layer-ReLU encoder's
    ``conv1`` into uniform and dynamic channels. Dynamic channels receive a
    GradNorm-weighted combination of task gradients; uniform channels receive
    normalized equal weighting. At epoch 0, task-specific heads receive the
    same normalized-EW update as standard GradNorm. Thereafter they receive
    their own unweighted task gradients, which reproduces standard GradNorm's
    task-head behavior. ``--hybridgradnorm_weight_task_heads`` remains an
    opt-in alternative that applies dynamic GradNorm weights to the heads.
    Consequently,
    ``hybridfraction=0`` is exactly normalized EW for every model parameter.

    The GradNorm objective measures gradient norms on the *dynamic channel
    subset only*: these are the parameters whose gradient mixture it controls.
    """

    def __init__(self):
        super(HybridGradNorm, self).__init__()

    def init_param(self):
        self.loss_scale = nn.Parameter(torch.ones(self.task_num, device=self.device))

    def _dynamic_task_grads(self, losses, params, masks):
        """Return masked vectors and unmasked shaped gradients for each task."""
        flat_per_task = []
        per_parameter = [[] for _ in params]
        for task_idx in range(self.task_num):
            grad_list = torch.autograd.grad(
                losses[task_idx],
                params,
                retain_graph=True,
                allow_unused=True,
            )
            flat_per_task.append(torch.cat([
                ((grad if grad is not None else param.new_zeros(param.shape)) * masks[param]).reshape(-1)
                for grad, param in zip(grad_list, params)
            ]))
            for param_idx, (grad, param) in enumerate(zip(grad_list, params)):
                per_parameter[param_idx].append(
                    grad if grad is not None else param.new_zeros(param.shape)
                )
        return torch.stack(flat_per_task, dim=0), [torch.stack(grads, dim=0) for grads in per_parameter]

    def _task_head_grads(self, losses):
        """Per-task gradients for every task-specific decoder parameter."""
        head_params = [param for param in self.decoders.parameters() if param.requires_grad]
        per_parameter = [[] for _ in head_params]
        for task_idx in range(self.task_num):
            grad_list = torch.autograd.grad(
                losses[task_idx], head_params, retain_graph=True, allow_unused=True,
            )
            for param_idx, (grad, param) in enumerate(zip(grad_list, head_params)):
                per_parameter[param_idx].append(
                    grad if grad is not None else param.new_zeros(param.shape)
                )
        return head_params, [torch.stack(grads, dim=0) for grads in per_parameter]

    def _blockwise_backward(self, losses, **kwargs):
        """HybridGradNorm for block-partitioned encoders such as SegNet.

        ``HybridHPS`` supplies whole parameter blocks here (unlike the
        channel-mask partition used by the two-layer encoder).  Uniform blocks
        receive normalized EW gradients; dynamic blocks receive GradNorm's
        weighted gradient.  At fraction one this reproduces normalized
        standard GradNorm for the encoder and its usual raw decoder-head
        gradients after the first epoch.
        """
        if self.rep_grad:
            raise ValueError('HybridGradNorm does not support representation gradients (rep_grad=True)')
        required_methods = ['get_dynamic_share_params', 'get_uniform_share_params']
        if any(not hasattr(self, name) for name in required_methods):
            raise ValueError('HybridGradNorm requires HybridHPS shared block groups')

        dynamic_params = [param for param in self.get_dynamic_share_params() if param.requires_grad]
        if len(dynamic_params) == 0:
            # fraction=0: exactly normalized EW for all model parameters.
            losses.mean().backward()
            return losses.new_full((self.task_num,), 1.0 / self.task_num).cpu().numpy()

        alpha = kwargs['alpha']
        normalize_weights = kwargs.get('normalize_weights', False)
        weight_task_heads = kwargs.get('weight_task_heads', False)
        weight_sum = 1.0 if normalize_weights else float(self.task_num)

        if self.epoch < 1:
            losses.mean().backward()
            return losses.new_full(
                (self.task_num,), 1.0 / self.task_num if normalize_weights else 1.0,
            ).cpu().numpy()

        dynamic_grads, per_parameter_task_grads = self._dynamic_task_grads(
            losses,
            dynamic_params,
            {param: torch.ones_like(param, dtype=torch.bool) for param in dynamic_params},
        )
        head_params, per_head_task_grads = self._task_head_grads(losses)
        reported_weight = weight_sum * F.softmax(self.loss_scale, dim=-1)

        grad_norms = torch.norm(reported_weight.unsqueeze(1) * dynamic_grads, p=2, dim=-1)
        mean_grad_norm = grad_norms.mean()
        relative_losses = torch.tensor(
            [losses[task_idx].item() / self.train_loss_buffer[task_idx, 0] for task_idx in range(self.task_num)],
            device=self.device,
        )
        inverse_training_rate = relative_losses / relative_losses.mean()
        target_grad_norms = (mean_grad_norm * inverse_training_rate.pow(alpha)).detach()
        (grad_norms - target_grad_norms).abs().sum().backward()

        # Start all blocks with normalized EW, then replace only dynamic
        # blocks. This keeps the uniform subset genuinely uniformly weighted.
        losses.mean().backward()
        for param, task_grads in zip(dynamic_params, per_parameter_task_grads):
            dynamic_grad = torch.sum(
                reported_weight.view(self.task_num, *([1] * (task_grads.ndim - 1))) * task_grads,
                dim=0,
            )
            param.grad.detach().copy_(dynamic_grad)

        # Match standard GradNorm's post-initialisation decoder behavior.
        head_weights = reported_weight if weight_task_heads else torch.ones_like(reported_weight)
        for param, task_grads in zip(head_params, per_head_task_grads):
            head_grad = torch.sum(
                head_weights.view(self.task_num, *([1] * (task_grads.ndim - 1))) * task_grads,
                dim=0,
            )
            param.grad.detach().copy_(head_grad)

        return reported_weight.detach().cpu().numpy()

    def backward(self, losses, **kwargs):
        if self.rep_grad:
            raise ValueError('HybridGradNorm does not support representation gradients (rep_grad=True)')
        if not hasattr(self, 'get_two_layer_relu_channel_partition'):
            return self._blockwise_backward(losses, **kwargs)

        if not self._is_two_layer_relu_encoder():
            return self._blockwise_backward(losses, **kwargs)

        partition = self.get_two_layer_relu_channel_partition()

        # No dynamic channels: this is exactly normalized EW, including heads.
        if partition['dynamic_channels'] == 0:
            losses.mean().backward()
            return losses.new_full((self.task_num,), 1.0 / self.task_num).cpu().numpy()

        alpha = kwargs['alpha']
        normalize_weights = kwargs.get('normalize_weights', False)
        weight_task_heads = kwargs.get('weight_task_heads', False)
        weight_sum = 1.0 if normalize_weights else float(self.task_num)
        params = list(partition['dynamic_parameter_masks'].keys())
        masks = partition['dynamic_parameter_masks']

        if self.epoch >= 1:
            loss_weight = weight_sum * F.softmax(self.loss_scale, dim=-1)
            dynamic_grads, per_parameter_task_grads = self._dynamic_task_grads(losses, params, masks)
            # Match standard GradNorm after its first epoch: each decoder head
            # keeps its own raw task gradient. The optional flag instead uses
            # the dynamic GradNorm mixture for decoder heads.
            head_params, per_head_task_grads = self._task_head_grads(losses)

            # Standard GradNorm, restricted to the dynamic channel subset.
            grad_norms = torch.norm(loss_weight.unsqueeze(1) * dynamic_grads, p=2, dim=-1)
            mean_grad_norm = grad_norms.mean()
            relative_losses = torch.tensor(
                [
                    losses[task_idx].item() / self.train_loss_buffer[task_idx, 0]
                    for task_idx in range(self.task_num)
                ],
                device=self.device,
            )
            inverse_training_rate = relative_losses / relative_losses.mean()
            target_grad_norms = (mean_grad_norm * inverse_training_rate.pow(alpha)).detach()
            gradnorm_objective = (grad_norms - target_grad_norms).abs().sum()
            gradnorm_objective.backward()
            reported_weight = loss_weight.detach().clone()
        else:
            reported_weight = losses.new_full(
                (self.task_num,),
                1.0 / self.task_num if normalize_weights else 1.0,
            )

        # Uniform EW is the starting gradient for static shared channels.
        losses.mean().backward()

        # Replace only selected conv1 entries with the GradNorm mixture.
        if self.epoch >= 1:
            for param, task_grads in zip(params, per_parameter_task_grads):
                dynamic_grad = torch.sum(
                    reported_weight.view(self.task_num, *([1] * (task_grads.ndim - 1))) * task_grads,
                    dim=0,
                )
                param.grad.detach().copy_(torch.where(masks[param], dynamic_grad, param.grad))

            # Decoder heads have no channel partition. Match standard GradNorm
            # by keeping their own raw task gradients after epoch 0; optionally
            # use the detached dynamic mixture instead.
            head_weights = reported_weight if weight_task_heads else torch.ones_like(reported_weight)
            for param, task_grads in zip(head_params, per_head_task_grads):
                head_grad = torch.sum(
                    head_weights.view(self.task_num, *([1] * (task_grads.ndim - 1))) * task_grads,
                    dim=0,
                )
                if param.grad is None:
                    param.grad = head_grad.detach().clone()
                else:
                    param.grad.detach().copy_(head_grad)

        return reported_weight.cpu().numpy()
