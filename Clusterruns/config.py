import argparse
import numpy as np
import torch

_parser = argparse.ArgumentParser(description='Configuration for LibMTL')
# general
_parser.add_argument('--mode', type=str, default='train', help='train, test')
_parser.add_argument('--seed', type=int, default=0, help='random seed')
_parser.add_argument('--gpu_id', default='0', type=str, help='gpu_id') 
_parser.add_argument('--weighting', type=str, default='EW',
    help='loss weighing strategies, option: EW, UW, GradNorm, GLS, RLW, \
        MGDA, PCGrad, GradVac, CAGrad, GradDrop, DWA, IMTL, DotProduct, \
        HybridDWA, HybridGradNorm, HybridRLWEW, HybridCAGrad, HybridDotProduct')
_parser.add_argument('--arch', type=str, default='HPS',
                    help='architecture for MTL, option: HPS, HybridHPS, MTAN')
_parser.add_argument('--rep_grad', action='store_true', default=False, 
                    help='computing gradient for representation or sharing parameters')
_parser.add_argument('--multi_input', action='store_true', default=False, 
                    help='whether each task has its own input data')
_parser.add_argument('--save_path', type=str, default=None, 
                    help='save path')
_parser.add_argument('--load_path', type=str, default=None, 
                    help='load ckpt path')
## optim
_parser.add_argument('--optim', type=str, default='adam',
                    help='optimizer for training, option: adam, sgd, adagrad, rmsprop')
_parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for all types of optim')
_parser.add_argument('--momentum', type=float, default=0.9, help='momentum for sgd')
_parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay for all types of optim')
## scheduler
_parser.add_argument('--scheduler', type=str, #default='step',
                    help='learning rate scheduler for training, option: step, cos, exp')
_parser.add_argument('--step_size', type=int, default=100, help='step size for StepLR')
_parser.add_argument('--gamma', type=float, default=0.5, help='gamma for StepLR')

# args for weighting
## DWA
_parser.add_argument('--T', type=float, default=2.0, help='T for DWA')
## MGDA
_parser.add_argument('--mgda_gn', default='none', type=str, 
                    help='type of gradient normalization for MGDA, option: l2, none, loss, loss+')
## GradVac
_parser.add_argument('--GradVac_beta', type=float, default=0.5, help='beta for GradVac')
_parser.add_argument('--GradVac_group_type', type=int, default=0, 
                    help='parameter granularity for GradVac (0: whole_model; 1: all_layer; 2: all_matrix)')
## GradNorm
_parser.add_argument('--alpha', type=float, default=1.5, help='alpha for GradNorm')
_parser.add_argument('--gradnorm_normalize_weights', action='store_true', default=False,
                    help='normalize GradNorm/HybridGradNorm task weights to sum to 1 instead of the standard number of tasks')
_parser.add_argument('--hybridgradnorm_weight_task_heads', action='store_true', default=False,
                    help='for HybridGradNorm, apply dynamic GradNorm weights to task-specific decoder heads instead of matching standard GradNorm\'s raw per-task head updates after epoch 0')
## GradDrop
_parser.add_argument('--leak', type=float, default=0.0, help='leak for GradDrop')
## CAGrad
_parser.add_argument('--calpha', type=float, default=0.5, help='calpha for CAGrad')
_parser.add_argument('--rescale', type=int, default=1, help='rescale for CAGrad')
## DotProduct
_parser.add_argument('--temperature', type=float, default=1.0,
                    help='softmax temperature for DotProduct weighting')
_parser.add_argument('--score_scaling', type=str, default='rms',
                    help='common score scaling for DotProduct, option: rms, none')
_parser.add_argument('--eps', type=float, default=1e-12,
                    help='numerical epsilon for DotProduct score scaling')
_parser.add_argument('--weight_update_frequency', type=int, default=1,
                    help='recompute DotProduct weights every n training steps')
_parser.add_argument('--gradient_scope', type=str, default='shared',
                    help='parameter scope for DotProduct gradients, option: shared, all')
_parser.add_argument('--meta_batch_size', type=int, default=None,
                    help='optional batch size cap for the held-out validation/meta batch used by DotProduct')
_parser.add_argument('--stage_meta_gradient', action='store_true', default=False,
                    help='compute DotProduct validation gradient before the training forward to reduce peak GPU memory')
## Nash_MTL
_parser.add_argument('--update_weights_every', type=int, default=1, help='update_weights_every for Nash_MTL')
_parser.add_argument('--optim_niter', type=int, default=20, help='optim_niter for Nash_MTL')
_parser.add_argument('--max_norm', type=float, default=1.0, help='max_norm for Nash_MTL')
## MoCo
_parser.add_argument('--MoCo_beta', type=float, default=0.5, help='MoCo_beta for MoCo')
_parser.add_argument('--MoCo_beta_sigma', type=float, default=0.5, help='MoCo_beta_sigma for MoCo')
_parser.add_argument('--MoCo_gamma', type=float, default=0.1, help='gamma for MoCo')
_parser.add_argument('--MoCo_gamma_sigma', type=float, default=0.5, help='MoCo_gamma_sigma for MoCo')
_parser.add_argument('--MoCo_rho', type=float, default=0, help='MoCo_rho for MoCo')

# args for architecture
## CGC
_parser.add_argument('--img_size', nargs='+', help='image size for CGC')
_parser.add_argument('--num_experts', nargs='+', help='the number of experts for sharing and task-specific')
## DSelect_k
_parser.add_argument('--num_nonzeros', type=int, default=2, help='num_nonzeros for DSelect-k')
_parser.add_argument('--kgamma', type=float, default=1.0, help='gamma for DSelect-k')
## HybridHPS
_parser.add_argument('--hybridfraction', type=float, default=0.0,
                    help='fraction of trailing shared blocks updated by the dynamic weighting method in HybridHPS')

LibMTL_args = _parser


def prepare_args(params):
    r"""Return the configuration of hyperparameters, optimizier, and learning rate scheduler.

    Args:
        params (argparse.Namespace): The command-line arguments.
    """
    kwargs = {'weight_args': {}, 'arch_args': {}}
    if params.weighting in ['EW', 'UW', 'GradNorm', 'GLS', 'RLW', 'MGDA', 'IMTL',
                            'PCGrad', 'GradVac', 'CAGrad', 'GradDrop', 'DWA',
                            'Nash_MTL', 'MoCo', 'Aligned_MTL', 'DotProduct',
                            'HybridDWA', 'HybridGradNorm', 'HybridRLWEW',
                            'HybridCAGrad', 'HybridDotProduct']:
        if params.weighting in ['DWA', 'HybridDWA']:
            if params.T is not None:
                kwargs['weight_args']['T'] = params.T
            else:
                raise ValueError('DWA needs keywaord T')
        elif params.weighting in ['GradNorm', 'HybridGradNorm']:
            if params.alpha is not None:
                kwargs['weight_args']['alpha'] = params.alpha
                if params.weighting in ['GradNorm', 'HybridGradNorm']:
                    kwargs['weight_args']['normalize_weights'] = params.gradnorm_normalize_weights
                if params.weighting == 'HybridGradNorm':
                    kwargs['weight_args']['weight_task_heads'] = params.hybridgradnorm_weight_task_heads
            else:
                raise ValueError('GradNorm needs keywaord alpha')
        elif params.weighting in ['MGDA']:
            if params.mgda_gn is not None:
                if params.mgda_gn in ['none', 'l2', 'loss', 'loss+']:
                    kwargs['weight_args']['mgda_gn'] = params.mgda_gn
                else:
                    raise ValueError('No support mgda_gn {} for MGDA'.format(params.mgda_gn)) 
            else:
                raise ValueError('MGDA needs keywaord mgda_gn')
        elif params.weighting in ['GradVac']:
            if params.GradVac_beta is not None:
                kwargs['weight_args']['GradVac_beta'] = params.GradVac_beta
                kwargs['weight_args']['GradVac_group_type'] = params.GradVac_group_type
            else:
                raise ValueError('GradVac needs keywaord beta')
        elif params.weighting in ['GradDrop']:
            if params.leak is not None:
                kwargs['weight_args']['leak'] = params.leak
            else:
                raise ValueError('GradDrop needs keywaord leak')
        elif params.weighting in ['CAGrad', 'HybridCAGrad']:
            if params.calpha is not None and params.rescale is not None:
                kwargs['weight_args']['calpha'] = params.calpha
                kwargs['weight_args']['rescale'] = params.rescale
            else:
                raise ValueError('CAGrad needs keywaord calpha and rescale')
        elif params.weighting in ['DotProduct', 'HybridDotProduct']:
            if params.temperature is None or params.temperature <= 0:
                raise ValueError('{} needs a positive temperature'.format(params.weighting))
            if params.score_scaling not in ['rms', 'none']:
                raise ValueError('No support score_scaling {} for {}'.format(params.score_scaling, params.weighting))
            if params.eps is None or params.eps <= 0:
                raise ValueError('{} needs a positive eps'.format(params.weighting))
            if params.weight_update_frequency is None or params.weight_update_frequency <= 0:
                raise ValueError('{} needs a positive weight_update_frequency'.format(params.weighting))
            if params.gradient_scope not in ['shared', 'all']:
                raise ValueError('No support gradient_scope {} for {}'.format(params.gradient_scope, params.weighting))
            if params.meta_batch_size is not None and params.meta_batch_size <= 0:
                raise ValueError('{} meta_batch_size must be positive when provided'.format(params.weighting))
            kwargs['weight_args']['temperature'] = params.temperature
            kwargs['weight_args']['score_scaling'] = params.score_scaling
            kwargs['weight_args']['eps'] = params.eps
            kwargs['weight_args']['weight_update_frequency'] = params.weight_update_frequency
            kwargs['weight_args']['gradient_scope'] = params.gradient_scope
            kwargs['weight_args']['meta_batch_size'] = params.meta_batch_size
            kwargs['weight_args']['stage_meta_gradient'] = params.stage_meta_gradient
        elif params.weighting in ['Nash_MTL']:
            if params.update_weights_every is not None and params.optim_niter is not None and params.max_norm is not None:
                kwargs['weight_args']['update_weights_every'] = params.update_weights_every
                kwargs['weight_args']['optim_niter'] = params.optim_niter
                kwargs['weight_args']['max_norm'] = params.max_norm
            else:
                raise ValueError('Nash_MTL needs update_weights_every, optim_niter, and max_norm')
        elif params.weighting in ['MoCo']:
            kwargs['weight_args']['MoCo_beta'] = params.MoCo_beta
            kwargs['weight_args']['MoCo_beta_sigma'] = params.MoCo_beta_sigma
            kwargs['weight_args']['MoCo_gamma'] = params.MoCo_gamma
            kwargs['weight_args']['MoCo_gamma_sigma'] = params.MoCo_gamma_sigma
            kwargs['weight_args']['MoCo_rho'] = params.MoCo_rho
    else:
        raise ValueError('No support weighting method {}'.format(params.weighting)) 
        
    if params.arch in ['HPS', 'HybridHPS', 'Cross_stitch', 'MTAN', 'CGC', 'PLE', 'MMoE', 'DSelect_k', 'DIY', 'LTB']:
        if params.arch in ['HybridHPS']:
            kwargs['arch_args']['hybridfraction'] = params.hybridfraction
        if params.arch in ['CGC', 'PLE', 'MMoE', 'DSelect_k']:
            kwargs['arch_args']['img_size'] = tuple(params.img_size)#np.array(params.img_size, dtype=int).prod()
            kwargs['arch_args']['num_experts'] = [int(num) for num in params.num_experts]
        if params.arch in ['DSelect_k']:
            kwargs['arch_args']['kgamma'] = params.kgamma
            kwargs['arch_args']['num_nonzeros'] = params.num_nonzeros
    else:
        raise ValueError('No support architecture method {}'.format(params.arch)) 
        
    if params.optim in ['adam', 'sgd', 'adagrad', 'rmsprop']:
        if params.optim == 'adam':
            optim_param = {'optim': 'adam', 'lr': params.lr, 'weight_decay': params.weight_decay}
        elif params.optim == 'sgd':
            optim_param = {'optim': 'sgd', 'lr': params.lr, 
                           'weight_decay': params.weight_decay, 'momentum': params.momentum}
    else:
        raise ValueError('No support optim method {}'.format(params.optim))
        
    if params.scheduler is not None:
        if params.scheduler in ['step', 'cos', 'exp']:
            if params.scheduler == 'step':
                scheduler_param = {'scheduler': 'step', 'step_size': params.step_size, 'gamma': params.gamma}
        else:
            raise ValueError('No support scheduler method {}'.format(params.scheduler))
    else:
        scheduler_param = None
    
    _display(params, kwargs, optim_param, scheduler_param)
    
    return kwargs, optim_param, scheduler_param

def _display(params, kwargs, optim_param, scheduler_param):
    print('='*40)
    print('General Configuration:')
    print('\tMode:', params.mode)
    print('\tWighting:', params.weighting)
    print('\tArchitecture:', params.arch)
    print('\tRep_Grad:', params.rep_grad)
    print('\tMulti_Input:', params.multi_input)
    print('\tSeed:', params.seed)
    print('\tSave Path:', params.save_path)
    print('\tLoad Path:', params.load_path)
    print('\tDevice: {}'.format('cuda:'+params.gpu_id if torch.cuda.is_available() else 'cpu'))
    for wa, p in zip(['weight_args', 'arch_args'], [params.weighting, params.arch]):
        if kwargs[wa] != {}:
            print('{} Configuration:'.format(p))
            for k, v in kwargs[wa].items():
                print('\t'+k+':', v)
    print('Optimizer Configuration:')
    for k, v in optim_param.items():
        print('\t'+k+':', v)
    if scheduler_param is not None:
        print('Scheduler Configuration:')
        for k, v in scheduler_param.items():
            print('\t'+k+':', v)
