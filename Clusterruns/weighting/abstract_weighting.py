import torch, sys, random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AbsWeighting(nn.Module):
    r"""An abstract class for weighting strategies.
    """
    def __init__(self):
        super(AbsWeighting, self).__init__()
        
    def init_param(self):
        r"""Define and initialize some trainable parameters required by specific weighting methods. 
        """
        pass

    def _resolve_params(self, params=None):
        if params is None:
            params = self.get_share_params()
        return [param for param in params if param.requires_grad]

    def _compute_grad_dim(self, params=None):
        params = self._resolve_params(params)
        grad_index = []
        for param in params:
            grad_index.append(param.data.numel())
        grad_dim = sum(grad_index)
        return params, grad_index, grad_dim

    def _grad2vec(self, params=None, grad_index=None, grad_dim=None):
        params = self._resolve_params(params)
        if grad_index is None or grad_dim is None:
            _, grad_index, grad_dim = self._compute_grad_dim(params)

        grad = torch.zeros(grad_dim).to(self.device)
        count = 0
        for param in params:
            if param.grad is not None:
                beg = 0 if count == 0 else sum(grad_index[:count])
                end = sum(grad_index[:(count+1)])
                grad[beg:end] = param.grad.data.view(-1)
            count += 1
        return grad

    def _compute_grad(self, losses, mode, rep_grad=False, params=None, zero_fn=None):
        '''
        mode: backward, autograd
        '''
        if not rep_grad:
            params, grad_index, grad_dim = self._compute_grad_dim(params)
            grads = torch.zeros(self.task_num, grad_dim).to(self.device)
            for tn in range(self.task_num):
                if mode == 'backward':
                    losses[tn].backward(retain_graph=True) if (tn+1) != self.task_num else losses[tn].backward()
                    grads[tn] = self._grad2vec(params=params, grad_index=grad_index, grad_dim=grad_dim)
                elif mode == 'autograd':
                    grad = list(torch.autograd.grad(losses[tn], params, retain_graph=True, allow_unused=True))
                    grads[tn] = torch.cat([
                        g.view(-1) if g is not None else param.new_zeros(param.numel())
                        for g, param in zip(grad, params)
                    ])
                else:
                    raise ValueError('No support {} mode for gradient computation'.format(mode))

                if zero_fn is None:
                    self.zero_grad_share_params()
                else:
                    zero_fn()
        else:
            if not isinstance(self.rep, dict):
                grads = torch.zeros(self.task_num, *self.rep.size()).to(self.device)
            else:
                grads = [torch.zeros(*self.rep[task].size()) for task in self.task_name]
            for tn, task in enumerate(self.task_name):
                if mode == 'backward':
                    losses[tn].backward(retain_graph=True) if (tn+1)!=self.task_num else losses[tn].backward()
                    grads[tn] = self.rep_tasks[task].grad.data.clone()
        return grads

    def _reset_grad(self, new_grads, params=None, grad_index=None):
        params = self._resolve_params(params)
        if grad_index is None:
            _, grad_index, _ = self._compute_grad_dim(params)

        count = 0
        for param in params:
            beg = 0 if count == 0 else sum(grad_index[:count])
            end = sum(grad_index[:(count+1)])
            reshaped_grad = new_grads[beg:end].contiguous().view(param.data.size()).data.clone()
            if param.grad is None:
                param.grad = reshaped_grad
            else:
                param.grad.data = reshaped_grad
            count += 1
            
    def _get_grads(self, losses, mode='backward', params=None, zero_fn=None):
        r"""This function is used to return the gradients of representations or shared parameters.

        If ``rep_grad`` is ``True``, it returns a list with two elements. The first element is \
        the gradients of the representations with the size of [task_num, batch_size, rep_size]. \
        The second element is the resized gradients with size of [task_num, -1], which means \
        the gradient of each task is resized as a vector.

        If ``rep_grad`` is ``False``, it returns the gradients of the shared parameters with size \
        of [task_num, -1], which means the gradient of each task is resized as a vector.
        """
        if self.rep_grad:
            per_grads = self._compute_grad(losses, mode, rep_grad=True)
            if not isinstance(self.rep, dict):
                grads = per_grads.reshape(self.task_num, self.rep.size()[0], -1).sum(1)
            else:
                try:
                    grads = torch.stack(per_grads).sum(1).view(self.task_num, -1)
                except:
                    raise ValueError('The representation dimensions of different tasks must be consistent')
            return [per_grads, grads]
        else:
            grads = self._compute_grad(losses, mode, params=params, zero_fn=zero_fn)
            return grads
        
    def _backward_new_grads(self, batch_weight, per_grads=None, grads=None, params=None):
        r"""This function is used to reset the gradients and make a backward.

        Args:
            batch_weight (torch.Tensor): A tensor with size of [task_num].
            per_grad (torch.Tensor): It is needed if ``rep_grad`` is True. The gradients of the representations.
            grads (torch.Tensor): It is needed if ``rep_grad`` is False. The gradients of the shared parameters. 
        """
        if self.rep_grad:
            if not isinstance(self.rep, dict):
                # transformed_grad = torch.einsum('i, i... -> ...', batch_weight, per_grads)
                transformed_grad = sum([batch_weight[i] * per_grads[i] for i in range(self.task_num)])
                self.rep.backward(transformed_grad)
            else:
                for tn, task in enumerate(self.task_name):
                    rg = True if (tn+1)!=self.task_num else False
                    self.rep[task].backward(batch_weight[tn]*per_grads[tn], retain_graph=rg)
        else:
            params, grad_index, _ = self._compute_grad_dim(params)
            new_grads = sum([batch_weight[i] * grads[i] for i in range(self.task_num)])
            self._reset_grad(new_grads, params=params, grad_index=grad_index)
    
    @property
    def backward(self, losses, **kwargs):
        r"""
        Args:
            losses (list): A list of losses of each task.
            kwargs (dict): A dictionary of hyperparameters of weighting methods.
        """
        pass
