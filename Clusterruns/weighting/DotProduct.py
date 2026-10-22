import numpy as np
import torch

from LibMTL.weighting.abstract_weighting import AbsWeighting


class DotProduct(AbsWeighting):
    r"""Dynamic task weighting based on train/validation gradient alignment.

    At each weighting update, the method computes each task's training gradient and
    the gradient of the equally weighted validation loss, scores each task by their
    dot product, and converts the scores to simplex-constrained weights with a
    temperature-scaled softmax.
    """

    def __init__(self):
        super(DotProduct, self).__init__()
        self.requires_meta_val = True

    def init_param(self):
        self._dot_product_step = 0
        self._cached_batch_weight = None
        self.latest_dot_product_scores = None
        self.latest_scaled_dot_product_scores = None
        self.latest_score_scale = None
        self.latest_val_losses = None
        self.latest_alignment_matrix = None
        self._staged_val_grad = None

    def stage_validation_gradient(self, val_losses, **kwargs):
        """Compute and retain only the average validation-gradient vector.

        This is mathematically identical for the DotProduct scores, but lets the
        trainer release the validation forward graph before constructing the
        training forward graph.  The optional full validation-to-training
        alignment matrix is intentionally unavailable in this memory-saving mode.
        """
        params = self._get_scope_params(kwargs['gradient_scope'])
        val_grad_list = torch.autograd.grad(
            val_losses.mean(), params, retain_graph=False, allow_unused=True
        )
        self._staged_val_grad = torch.cat([
            grad.reshape(-1) if grad is not None else param.new_zeros(param.numel())
            for grad, param in zip(val_grad_list, params)
        ]).detach()
        self.latest_val_losses = val_losses.detach().clone()
        self.latest_alignment_matrix = None

    def _get_scope_params(self, gradient_scope):
        if gradient_scope == 'shared':
            params = self.get_share_params()
        elif gradient_scope == 'all':
            params = self.parameters()
        else:
            raise ValueError('No support gradient_scope {} for DotProduct'.format(gradient_scope))
        params = [param for param in params if param.requires_grad]
        if len(params) == 0:
            raise ValueError('DotProduct could not find any trainable parameters in scope {}'.format(gradient_scope))
        return params

    def backward(self, losses, **kwargs):
        if self.rep_grad:
            raise ValueError('DotProduct does not support representation gradients (rep_grad=True)')

        temperature = kwargs['temperature']
        score_scaling = kwargs['score_scaling']
        eps = kwargs['eps']
        weight_update_frequency = kwargs['weight_update_frequency']
        gradient_scope = kwargs['gradient_scope']
        val_losses = kwargs.get('val_losses', None)
        staged_val_grad = self._staged_val_grad

        if val_losses is None and staged_val_grad is None:
            raise ValueError('DotProduct requires per-task validation losses passed as val_losses')

        params = self._get_scope_params(gradient_scope)
        train_losses = losses
        if val_losses is not None:
            self.latest_val_losses = val_losses.detach().clone()

        recompute = (
            self._cached_batch_weight is None
            or self._dot_product_step % weight_update_frequency == 0
        )

        if recompute:
            train_grads = self._get_grads(train_losses, mode='autograd', params=params)
            if staged_val_grad is not None:
                val_grad = staged_val_grad
                self._staged_val_grad = None
            else:
                self.latest_val_losses = val_losses.detach().clone()
                val_grads = []
                for task_idx in range(self.task_num):
                    val_grad_list = torch.autograd.grad(
                        val_losses[task_idx],
                        params,
                        retain_graph=(task_idx + 1) != self.task_num,
                        allow_unused=True,
                    )
                    val_grad = torch.cat([
                        grad.view(-1) if grad is not None else param.new_zeros(param.numel())
                        for grad, param in zip(val_grad_list, params)
                    ])
                    val_grads.append(val_grad)
                val_grads = torch.stack(val_grads, dim=0)
                val_grad = val_grads.mean(dim=0)
                self.latest_alignment_matrix = torch.matmul(val_grads, train_grads.transpose(0, 1)).detach().clone()

            scores = torch.matmul(train_grads, val_grad)
            if score_scaling == 'rms':
                common_scale = torch.sqrt(torch.mean(scores.pow(2))) + eps
            elif score_scaling == 'none':
                common_scale = scores.new_tensor(1.0)
            else:
                raise ValueError('No support score_scaling {} for DotProduct'.format(score_scaling))

            scaled_scores = scores / common_scale
            stable_logits = (scaled_scores / temperature) - (scaled_scores / temperature).max()
            batch_weight = torch.softmax(stable_logits, dim=0).detach()

            self._cached_batch_weight = batch_weight
            self.latest_dot_product_scores = scores.detach().clone()
            self.latest_scaled_dot_product_scores = scaled_scores.detach().clone()
            self.latest_score_scale = common_scale.detach().clone()
        elif self.latest_dot_product_scores is None:
            self.latest_dot_product_scores = train_losses.new_zeros(self.task_num)
            self.latest_scaled_dot_product_scores = train_losses.new_zeros(self.task_num)
            self.latest_score_scale = train_losses.new_tensor(1.0)
            self.latest_alignment_matrix = train_losses.new_zeros((self.task_num, self.task_num))

        weighted_loss = torch.sum(self._cached_batch_weight * train_losses)
        weighted_loss.backward()

        self._dot_product_step += 1
        return self._cached_batch_weight.detach().cpu().numpy()
