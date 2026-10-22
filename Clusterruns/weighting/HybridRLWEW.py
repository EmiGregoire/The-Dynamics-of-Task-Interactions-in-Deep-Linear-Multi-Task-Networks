import torch
import torch.nn.functional as F
import numpy as np

from LibMTL.weighting.abstract_weighting import AbsWeighting


class HybridRLWEW(AbsWeighting):
    r"""Hybrid Random Loss Weighting + Equal Weighting (HybridRLWEW).

    This method applies Equal Weighting (EW) to the uniform subset of shared
    parameters and Random Loss Weighting (RLW) to the dynamic subset.
    """
    def __init__(self):
        super(HybridRLWEW, self).__init__()

    def backward(self, losses, **kwargs):
        if not hasattr(self, 'get_uniform_share_params'):
            raise ValueError('HybridRLWEW requires an architecture exposing uniform shared parameter groups')

        uniform_params = list(self.get_uniform_share_params())

        dynamic_weight = F.softmax(torch.randn(self.task_num), dim=-1).to(self.device)
        loss = torch.mul(losses, dynamic_weight).sum()
        loss.backward(retain_graph=True)

        if len(uniform_params) > 0:
            uniform_weight = torch.ones_like(losses).to(self.device)
            uniform_grads = self._get_grads(
                losses,
                mode='autograd',
                params=uniform_params
            )
            self._backward_new_grads(uniform_weight, grads=uniform_grads, params=uniform_params)

        return dynamic_weight.detach().cpu().numpy()
