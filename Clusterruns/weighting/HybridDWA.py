import torch
import torch.nn.functional as F
import numpy as np

from LibMTL.weighting.abstract_weighting import AbsWeighting


class HybridDWA(AbsWeighting):
    r"""Hybrid Dynamic Weight Average (HybridDWA).

    This method applies Equal Weighting (EW) to the uniform subset of shared
    parameters and Dynamic Weight Average (DWA) to the dynamic subset.
    """
    def __init__(self):
        super(HybridDWA, self).__init__()

    def _get_uniform_loss_weight(self, losses):
        return torch.ones_like(losses).to(self.device)

    def _get_dynamic_loss_weight(self, losses, T):
        if self.epoch > 1:
            w_i = torch.Tensor(
                self.train_loss_buffer[:, self.epoch-1] / self.train_loss_buffer[:, self.epoch-2]
            ).to(self.device)
            return self.task_num * F.softmax(w_i / T, dim=-1)
        return torch.ones_like(losses).to(self.device)

    def backward(self, losses, **kwargs):
        if not hasattr(self, 'get_uniform_share_params'):
            raise ValueError('HybridDWA requires an architecture exposing uniform shared parameter groups')

        T = kwargs['T']
        uniform_params = list(self.get_uniform_share_params())

        dynamic_weight = self._get_dynamic_loss_weight(losses, T)
        loss = torch.mul(losses, dynamic_weight).sum()
        loss.backward(retain_graph=True)

        if len(uniform_params) > 0:
            uniform_weight = self._get_uniform_loss_weight(losses)
            uniform_grads = self._get_grads(
                losses,
                mode='autograd',
                params=uniform_params
            )
            self._backward_new_grads(uniform_weight, grads=uniform_grads, params=uniform_params)

        return dynamic_weight.detach().cpu().numpy()
