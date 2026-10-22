import torch

from LibMTL.weighting.DotProduct import DotProduct


class HybridDotProduct(DotProduct):
    r"""Channel-wise hybrid DotProduct weighting for the two-layer NYUv2 model.

    ``HybridHPS`` partitions the output filters of the shared ``conv1`` into
    a uniform group and a dynamic group.  This method applies normalized equal
    weighting to the uniform filters and DotProduct weighting to the dynamic
    filters.  Task-specific heads always receive normalized-EW gradients, so
    ``hybridfraction=0`` is exactly normalized EW for every parameter.
    """

    def init_param(self):
        super(HybridDotProduct, self).init_param()
        partition = self.get_two_layer_relu_channel_partition()
        # Avoid held-out validation work entirely when the method is exactly
        # normalized EW (no dynamic output channels).
        self.requires_meta_val = partition['dynamic_channels'] > 0

    def _dynamic_weights(self, losses, **kwargs):
        """Compute/cache the DotProduct simplex weights on shared parameters."""
        temperature = kwargs['temperature']
        score_scaling = kwargs['score_scaling']
        eps = kwargs['eps']
        update_frequency = kwargs['weight_update_frequency']
        gradient_scope = kwargs['gradient_scope']
        val_losses = kwargs.get('val_losses', None)
        staged_val_grad = self._staged_val_grad

        if gradient_scope != 'shared':
            raise ValueError('HybridDotProduct currently supports gradient_scope=shared only')
        if val_losses is None and staged_val_grad is None:
            raise ValueError('HybridDotProduct requires per-task validation losses')

        params = self._get_scope_params('shared')
        recompute = (
            self._cached_batch_weight is None
            or self._dot_product_step % update_frequency == 0
        )

        if recompute:
            train_grads = self._get_grads(losses, mode='autograd', params=params)
            if staged_val_grad is not None:
                val_grad = staged_val_grad
                self._staged_val_grad = None
            else:
                self.latest_val_losses = val_losses.detach().clone()
                val_grads = []
                for task_idx in range(self.task_num):
                    grad_list = torch.autograd.grad(
                        val_losses[task_idx],
                        params,
                        retain_graph=(task_idx + 1) != self.task_num,
                        allow_unused=True,
                    )
                    val_grads.append(torch.cat([
                        grad.reshape(-1) if grad is not None else param.new_zeros(param.numel())
                        for grad, param in zip(grad_list, params)
                    ]))
                val_grads = torch.stack(val_grads)
                val_grad = val_grads.mean(dim=0)
                self.latest_alignment_matrix = torch.matmul(
                    val_grads, train_grads.transpose(0, 1)
                ).detach().clone()

            scores = torch.matmul(train_grads, val_grad)
            if score_scaling == 'rms':
                score_scale = torch.sqrt(torch.mean(scores.pow(2))) + eps
            elif score_scaling == 'none':
                score_scale = scores.new_tensor(1.0)
            else:
                raise ValueError('No support score_scaling {} for HybridDotProduct'.format(score_scaling))

            scaled_scores = scores / score_scale
            logits = scaled_scores / temperature
            weights = torch.softmax(logits - logits.max(), dim=0).detach()
            self._cached_batch_weight = weights
            self.latest_dot_product_scores = scores.detach().clone()
            self.latest_scaled_dot_product_scores = scaled_scores.detach().clone()
            self.latest_score_scale = score_scale.detach().clone()
        elif self.latest_dot_product_scores is None:
            self.latest_dot_product_scores = losses.new_zeros(self.task_num)
            self.latest_scaled_dot_product_scores = losses.new_zeros(self.task_num)
            self.latest_score_scale = losses.new_tensor(1.0)
            self.latest_alignment_matrix = losses.new_zeros((self.task_num, self.task_num))

        return self._cached_batch_weight

    def _per_task_parameter_grads(self, losses, params):
        per_task = [[] for _ in params]
        for task_idx in range(self.task_num):
            grad_list = torch.autograd.grad(
                losses[task_idx],
                params,
                retain_graph=True,
                allow_unused=True,
            )
            for param_idx, (grad, param) in enumerate(zip(grad_list, params)):
                per_task[param_idx].append(
                    grad if grad is not None else param.new_zeros(param.shape)
                )
        return [torch.stack(task_grads, dim=0) for task_grads in per_task]

    def backward(self, losses, **kwargs):
        if self.rep_grad:
            raise ValueError('HybridDotProduct does not support representation gradients (rep_grad=True)')

        partition = self.get_two_layer_relu_channel_partition()
        if partition['dynamic_channels'] == 0:
            # This is deliberately identical to normalized EW, including the
            # task-specific heads.
            losses.mean().backward()
            self._dot_product_step += 1
            return losses.new_full((self.task_num,), 1.0 / self.task_num).cpu().numpy()

        weights = self._dynamic_weights(losses, **kwargs)
        params = list(partition['dynamic_parameter_masks'].keys())
        task_grads = self._per_task_parameter_grads(losses, params)

        # Populate all task-head gradients with normalized EW.  The shared
        # conv1 gradients are replaced below by the channel-wise hybrid rule.
        losses.mean().backward()

        for param, grads in zip(params, task_grads):
            uniform_grad = grads.mean(dim=0)
            dynamic_grad = torch.sum(weights.view(self.task_num, *([1] * (grads.ndim - 1))) * grads, dim=0)
            dynamic_mask = partition['dynamic_parameter_masks'][param]
            hybrid_grad = torch.where(dynamic_mask, dynamic_grad, uniform_grad)
            param.grad.detach().copy_(hybrid_grad)

        self._dot_product_step += 1
        return weights.detach().cpu().numpy()
