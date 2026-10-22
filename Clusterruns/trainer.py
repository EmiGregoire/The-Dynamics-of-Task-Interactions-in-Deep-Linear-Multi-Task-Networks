import csv
import torch, os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL._record import _PerformanceMeter
from LibMTL.utils import count_parameters

class Trainer(nn.Module):
    r'''A Multi-Task Learning Trainer.

    This is a unified and extensible training framework for multi-task learning. 

    Args:
        task_dict (dict): A dictionary of name-information pairs of type (:class:`str`, :class:`dict`). \
                            The sub-dictionary for each task has four entries whose keywords are named **metrics**, \
                            **metrics_fn**, **loss_fn**, **weight** and each of them corresponds to a :class:`list`.
                            The list of **metrics** has ``m`` strings, repersenting the name of ``m`` metrics \
                            for this task. The list of **metrics_fn** has two elements, i.e., the updating and score \
                            functions, meaning how to update thoes objectives in the training process and obtain the final \
                            scores, respectively. The list of **loss_fn** has ``m`` loss functions corresponding to each \
                            metric. The list of **weight** has ``m`` binary integers corresponding to each \
                            metric, where ``1`` means the higher the score is, the better the performance, \
                            ``0`` means the opposite.                           
        weighting (class): A weighting strategy class based on :class:`LibMTL.weighting.abstract_weighting.AbsWeighting`.
        architecture (class): An architecture class based on :class:`LibMTL.architecture.abstract_arch.AbsArchitecture`.
        encoder_class (class): A neural network class.
        decoders (dict): A dictionary of name-decoder pairs of type (:class:`str`, :class:`torch.nn.Module`).
        rep_grad (bool): If ``True``, the gradient of the representation for each task can be computed.
        multi_input (bool): Is ``True`` if each task has its own input data, otherwise is ``False``. 
        optim_param (dict): A dictionary of configurations for the optimizier.
        scheduler_param (dict): A dictionary of configurations for learning rate scheduler. \
                                 Set it to ``None`` if you do not use a learning rate scheduler.
        task_diagnostics_path (str, default=None): If provided, write per-batch task losses, gradient norms, \
                                 pairwise cosine similarities, and magnitude imbalances to a CSV file.
        task_diagnostics_every (int, default=1): Log diagnostics every ``n`` training batches.
        kwargs (dict): A dictionary of hyperparameters of weighting and architecture methods.

    .. note::
            It is recommended to use :func:`LibMTL.config.prepare_args` to return the dictionaries of ``optim_param``, \
            ``scheduler_param``, and ``kwargs``.

    Examples::
        
        import torch.nn as nn
        from LibMTL import Trainer
        from LibMTL.loss import CE_loss_fn
        from LibMTL.metrics import acc_update_fun, acc_score_fun
        from LibMTL.weighting import EW
        from LibMTL.architecture import HPS
        from LibMTL.model import ResNet18
        from LibMTL.config import prepare_args

        task_dict = {'A': {'metrics': ['Acc'],
                           'metrics_fn': [acc_update_fun, acc_score_fun],
                           'loss_fn': [CE_loss_fn],
                           'weight': [1]}}
        
        decoders = {'A': nn.Linear(512, 31)}
        
        # You can use command-line arguments and return configurations by ``prepare_args``.
        # kwargs, optim_param, scheduler_param = prepare_args(params)
        optim_param = {'optim': 'adam', 'lr': 1e-3, 'weight_decay': 1e-4}
        scheduler_param = {'scheduler': 'step'}
        kwargs = {'weight_args': {}, 'arch_args': {}}

        trainer = Trainer(task_dict=task_dict,
                          weighting=EW,
                          architecture=HPS,
                          encoder_class=ResNet18,
                          decoders=decoders,
                          rep_grad=False,
                          multi_input=False,
                          optim_param=optim_param,
                          scheduler_param=scheduler_param,
                          **kwargs)

    '''
    def __init__(self, task_dict, weighting, architecture, encoder_class, decoders, 
                 rep_grad, multi_input, optim_param, scheduler_param, 
                 save_path=None, load_path=None, task_diagnostics_path=None,
                 task_diagnostics_every=1, epoch_metrics_path=None, test_every=1,
                 save_best=True, freeze_bn=False, **kwargs):
        super(Trainer, self).__init__()
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Used device: {}'.format(self.device))
        self.kwargs = kwargs
        self.task_dict = task_dict
        self.task_num = len(task_dict)
        self.task_name = list(task_dict.keys())
        self.rep_grad = rep_grad
        self.multi_input = multi_input
        self.scheduler_param = scheduler_param
        self.save_path = save_path
        self.save_best = save_best
        self.freeze_bn = freeze_bn
        self.load_path = load_path
        self.task_diagnostics_path = task_diagnostics_path
        self.task_diagnostics_every = max(1, task_diagnostics_every)
        self.task_diagnostics_header = self._build_task_diagnostics_header()
        self.epoch_metrics_path = epoch_metrics_path
        self.epoch_metrics_header = self._build_epoch_metrics_header()
        self.test_every = max(1, test_every)

        self._prepare_model(weighting, architecture, encoder_class, decoders)
        self._prepare_optimizer(optim_param, scheduler_param)
        if self.freeze_bn:
            self._freeze_batch_norm()
        self.requires_meta_val = getattr(self.model, 'requires_meta_val', False)
        
        self.meter = _PerformanceMeter(self.task_dict, self.multi_input)
        
    def _prepare_model(self, weighting, architecture, encoder_class, decoders):
        
        class MTLmodel(architecture, weighting):
            def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device, kwargs):
                super(MTLmodel, self).__init__(task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs)
                self.init_param()
                
        self.model = MTLmodel(task_name=self.task_name, 
                              encoder_class=encoder_class, 
                              decoders=decoders, 
                              rep_grad=self.rep_grad, 
                              multi_input=self.multi_input,
                              device=self.device,
                              kwargs=self.kwargs['arch_args']).to(self.device)
        if self.load_path is not None:
            if os.path.isdir(self.load_path):
                self.load_path = os.path.join(self.load_path, 'best.pt')
            self.model.load_state_dict(torch.load(self.load_path, map_location=self.device), strict=False)
            print('Load Model from - {}'.format(self.load_path))
        count_parameters(self.model)
        
    def _freeze_batch_norm(self):
        for module in self.model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

    def _prepare_optimizer(self, optim_param, scheduler_param):
        optim_dict = {
                'sgd': torch.optim.SGD,
                'adam': torch.optim.Adam,
                'adagrad': torch.optim.Adagrad,
                'rmsprop': torch.optim.RMSprop,
            }
        scheduler_dict = {
                'exp': torch.optim.lr_scheduler.ExponentialLR,
                'step': torch.optim.lr_scheduler.StepLR,
                'cos': torch.optim.lr_scheduler.CosineAnnealingLR,
                'reduce': torch.optim.lr_scheduler.ReduceLROnPlateau,
            }
        optim_arg = {k: v for k, v in optim_param.items() if k != 'optim'}
        self.optimizer = optim_dict[optim_param['optim']](self.model.parameters(), **optim_arg)
        if scheduler_param is not None:
            scheduler_arg = {k: v for k, v in scheduler_param.items() if k != 'scheduler'}
            self.scheduler = scheduler_dict[scheduler_param['scheduler']](self.optimizer, **scheduler_arg)
        else:
            self.scheduler = None

    def _process_data(self, loader):
        try:
            data, label = next(loader[1])
        except:
            loader[1] = iter(loader[0])
            data, label = next(loader[1])
        data = data.to(self.device, non_blocking=True)
        if not self.multi_input:
            for task in self.task_name:
                label[task] = label[task].to(self.device, non_blocking=True)
        else:
            label = label.to(self.device, non_blocking=True)
        return data, label
    
    def process_preds(self, preds, task_name=None):
        r'''The processing of prediction for each task. 

        - The default is no processing. If necessary, you can rewrite this function. 
        - If ``multi_input`` is ``True``, ``task_name`` is valid and ``preds`` with type :class:`torch.Tensor` is the prediction of this task.
        - otherwise, ``task_name`` is invalid and ``preds`` is a :class:`dict` of name-prediction pairs of all tasks.

        Args:
            preds (dict or torch.Tensor): The prediction of ``task_name`` or all tasks.
            task_name (str): The string of task name.
        '''
        return preds

    def _compute_loss(self, preds, gts, task_name=None):
        if not self.multi_input:
            train_losses = torch.zeros(self.task_num).to(self.device)
            for tn, task in enumerate(self.task_name):
                train_losses[tn] = self.meter.losses[task]._update_loss(preds[task], gts[task])
        else:
            train_losses = self.meter.losses[task_name]._update_loss(preds, gts)
        return train_losses

    def _compute_loss_no_update(self, preds, gts, task_name=None):
        if not self.multi_input:
            losses = torch.zeros(self.task_num).to(self.device)
            for tn, task in enumerate(self.task_name):
                losses[tn] = self.meter.losses[task].compute_loss(preds[task], gts[task])
        else:
            losses = self.meter.losses[task_name].compute_loss(preds, gts)
        return losses
        
    def _prepare_dataloaders(self, dataloaders):
        if not self.multi_input:
            loader = [dataloaders, iter(dataloaders)]
            return loader, len(dataloaders)
        else:
            loader = {}
            batch_num = []
            for task in self.task_name:
                loader[task] = [dataloaders[task], iter(dataloaders[task])]
                batch_num.append(len(dataloaders[task]))
            return loader, batch_num

    def _build_task_diagnostics_header(self):
        header = ['epoch', 'batch', 'global_step']
        header.extend(['loss_{}'.format(task) for task in self.task_name])
        header.extend(['val_loss_{}'.format(task) for task in self.task_name])
        header.append('score_scale')
        header.extend(['weight_{}'.format(task) for task in self.task_name])
        header.extend(['score_{}'.format(task) for task in self.task_name])
        header.extend(['scaled_score_{}'.format(task) for task in self.task_name])
        for val_task in self.task_name:
            for train_task in self.task_name:
                header.append('align_val_{}_train_{}'.format(val_task, train_task))
        header.extend(['grad_norm_{}'.format(task) for task in self.task_name])
        header.append('grad_norm_imbalance')
        for tn_i, task_i in enumerate(self.task_name):
            for tn_j in range(tn_i + 1, self.task_num):
                task_j = self.task_name[tn_j]
                header.append('cosine_{}_{}'.format(task_i, task_j))
                header.append('magnitude_imbalance_{}_{}'.format(task_i, task_j))
        return header

    def _build_epoch_metrics_header(self):
        header = ['mode', 'epoch']
        for task in self.task_name:
            header.append('loss_{}'.format(task))
            for metric_name in self.task_dict[task]['metrics']:
                header.append('{}_{}'.format(task, metric_name))
        header.extend(['weight_{}'.format(task) for task in self.task_name])
        return header

    def _prepare_task_diagnostics_file(self):
        if self.task_diagnostics_path is None:
            return
        diag_dir = os.path.dirname(self.task_diagnostics_path)
        if diag_dir:
            os.makedirs(diag_dir, exist_ok=True)
        with open(self.task_diagnostics_path, 'w', newline='') as diag_file:
            csv.writer(diag_file).writerow(self.task_diagnostics_header)

    def _prepare_epoch_metrics_file(self):
        if self.epoch_metrics_path is None:
            return
        metrics_dir = os.path.dirname(self.epoch_metrics_path)
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)
        if not os.path.exists(self.epoch_metrics_path):
            with open(self.epoch_metrics_path, 'w', newline='') as metrics_file:
                csv.writer(metrics_file).writerow(self.epoch_metrics_header)

    def _slice_structure(self, item, max_items):
        if max_items is None:
            return item
        if torch.is_tensor(item):
            return item[:max_items]
        if isinstance(item, dict):
            return {key: self._slice_structure(value, max_items) for key, value in item.items()}
        if isinstance(item, list):
            return [self._slice_structure(value, max_items) for value in item]
        if isinstance(item, tuple):
            return tuple(self._slice_structure(value, max_items) for value in item)
        return item

    def _compute_meta_validation_losses(self, loader, meta_batch_size=None):
        prev_training = self.model.training
        self.model.eval()
        try:
            if not self.multi_input:
                val_inputs, val_gts = self._process_data(loader)
                val_inputs = self._slice_structure(val_inputs, meta_batch_size)
                val_gts = self._slice_structure(val_gts, meta_batch_size)
                val_preds = self.model(val_inputs)
                val_preds = self.process_preds(val_preds)
                val_losses = self._compute_loss_no_update(val_preds, val_gts)
            else:
                val_losses = torch.zeros(self.task_num).to(self.device)
                for tn, task in enumerate(self.task_name):
                    val_input, val_gt = self._process_data(loader[task])
                    val_input = self._slice_structure(val_input, meta_batch_size)
                    val_gt = self._slice_structure(val_gt, meta_batch_size)
                    val_pred = self.model(val_input, task)
                    val_pred = val_pred[task]
                    val_pred = self.process_preds(val_pred, task)
                    val_losses[tn] = self._compute_loss_no_update(val_pred, val_gt, task)
        finally:
            self.model.train(prev_training)
            if prev_training and self.freeze_bn:
                self._freeze_batch_norm()
        return val_losses

    def _compute_task_diagnostics(self, train_losses):
        grad_vec = []
        if self.rep_grad:
            for tn, task in enumerate(self.task_name):
                rep_grad = torch.autograd.grad(
                    train_losses[tn],
                    self.model.rep_tasks[task],
                    retain_graph=True,
                    allow_unused=True
                )[0]
                if rep_grad is None:
                    rep_grad = self.model.rep_tasks[task].new_zeros(self.model.rep_tasks[task].numel())
                else:
                    rep_grad = rep_grad.reshape(-1)
                grad_vec.append(rep_grad.detach())
        else:
            share_params = [param for param in self.model.get_share_params() if param.requires_grad]
            if len(share_params) == 0:
                return None
            for tn in range(self.task_num):
                task_grads = torch.autograd.grad(train_losses[tn], share_params, retain_graph=True, allow_unused=True)
                flat_grad = torch.cat([
                    grad.reshape(-1) if grad is not None else param.new_zeros(param.numel())
                    for grad, param in zip(task_grads, share_params)
                ])
                grad_vec.append(flat_grad.detach())
        grad_vec = torch.stack(grad_vec, dim=0)
        grad_norms = grad_vec.norm(p=2, dim=1)
        return {
            'losses': train_losses.detach().cpu(),
            'grad_vec': grad_vec,
            'grad_norms': grad_norms,
        }

    def _log_task_diagnostics(self, epoch, batch_index, global_step, diagnostics):
        if self.task_diagnostics_path is None or diagnostics is None:
            return

        losses = diagnostics['losses'].tolist()
        val_losses = getattr(self, '_current_meta_val_losses', None)
        if val_losses is None:
            val_losses = [np.nan] * self.task_num
        else:
            val_losses = val_losses
        scores = getattr(self, '_current_dot_product_scores', None)
        if scores is None:
            scores = [np.nan] * self.task_num
        scaled_scores = getattr(self, '_current_scaled_dot_product_scores', None)
        if scaled_scores is None:
            scaled_scores = [np.nan] * self.task_num
        score_scale = getattr(self, '_current_score_scale', np.nan)
        alignment_matrix = getattr(self, '_current_alignment_matrix', None)
        if alignment_matrix is None:
            alignment_values = [np.nan] * (self.task_num * self.task_num)
        else:
            alignment_values = alignment_matrix
        grad_norms = diagnostics['grad_norms'].detach().cpu().tolist()
        max_norm = max(grad_norms) if len(grad_norms) > 0 else 0.0
        min_norm = min(grad_norms) if len(grad_norms) > 0 else 0.0
        row = [epoch, batch_index, global_step]
        row.extend(losses)
        row.extend(val_losses)
        row.append(score_scale)
        row.extend(self._current_batch_weights)
        row.extend(scores)
        row.extend(scaled_scores)
        row.extend(alignment_values)
        row.extend(grad_norms)
        row.append(max_norm / (min_norm + 1e-12))

        grad_vec = diagnostics['grad_vec']
        for tn_i in range(self.task_num):
            for tn_j in range(tn_i + 1, self.task_num):
                cosine = F.cosine_similarity(
                    grad_vec[tn_i].unsqueeze(0),
                    grad_vec[tn_j].unsqueeze(0),
                    dim=1,
                    eps=1e-12
                ).item()
                norm_i = grad_norms[tn_i]
                norm_j = grad_norms[tn_j]
                imbalance = max(norm_i, norm_j) / (min(norm_i, norm_j) + 1e-12)
                row.extend([cosine, imbalance])

        with open(self.task_diagnostics_path, 'a', newline='') as diag_file:
            csv.writer(diag_file).writerow(row)

    def _log_epoch_metrics(self, mode, epoch):
        if self.epoch_metrics_path is None:
            return

        self._prepare_epoch_metrics_file()
        row = [mode, epoch]
        for tn, task in enumerate(self.task_name):
            row.append(self.meter.loss_item[tn])
            row.extend(self.meter.results[task])
        if mode == 'train' and hasattr(self, 'batch_weight'):
            for task_weights in self.batch_weight[:, epoch, :]:
                valid_weights = task_weights[~np.isnan(task_weights)]
                row.append(float(np.mean(valid_weights)) if len(valid_weights) else np.nan)
        else:
            row.extend([np.nan] * self.task_num)

        with open(self.epoch_metrics_path, 'a', newline='') as metrics_file:
            csv.writer(metrics_file).writerow(row)

    def train(self, train_dataloaders, test_dataloaders, epochs, 
              val_dataloaders=None, return_weight=False):
        r'''The training process of multi-task learning.

        Args:
            train_dataloaders (dict or torch.utils.data.DataLoader): The dataloaders used for training. \
                            If ``multi_input`` is ``True``, it is a dictionary of name-dataloader pairs. \
                            Otherwise, it is a single dataloader which returns data and a dictionary \
                            of name-label pairs in each iteration.

            test_dataloaders (dict or torch.utils.data.DataLoader): The dataloaders used for the validation or testing. \
                            The same structure with ``train_dataloaders``.
            epochs (int): The total training epochs.
            return_weight (bool): if ``True``, the loss weights will be returned.
        '''
        train_loader, train_batch = self._prepare_dataloaders(train_dataloaders)
        train_batch = max(train_batch) if self.multi_input else train_batch
        meta_loader = None
        if self.requires_meta_val:
            if val_dataloaders is None:
                raise ValueError(
                    'Weighting method {} requires held-out validation dataloaders during training. '
                    'Please pass val_dataloaders and keep that split held out.'.format(
                        self.model.__class__.__bases__[1].__name__
                    )
                )
            meta_loader, _ = self._prepare_dataloaders(val_dataloaders)
        
        self.batch_weight = np.full([self.task_num, epochs, train_batch], np.nan)
        self.model.train_loss_buffer = np.zeros([self.task_num, epochs])
        self.model.epochs = epochs
        self._prepare_task_diagnostics_file()
        self._prepare_epoch_metrics_file()
        initial_eval_logged = False
        if val_dataloaders is not None:
            self.meter.has_val = True
            self.test(val_dataloaders, 0, mode='val')
            initial_eval_logged = True
        if test_dataloaders is not None:
            self.test(test_dataloaders, 0, mode='test')
            initial_eval_logged = True
        for epoch in range(epochs):
            self.model.epoch = epoch
            self.model.train()
            if self.freeze_bn:
                self._freeze_batch_norm()
            self.meter.record_time('begin')
            for batch_index in range(train_batch):
                global_step = epoch * train_batch + batch_index
                weight_args = dict(self.kwargs['weight_args'])
                meta_gradient_staged = False
                if self.requires_meta_val and weight_args.get('stage_meta_gradient', False):
                    # Compute g_val before creating the training graph.  DotProduct
                    # retains only this detached vector, so the validation graph is
                    # released before the training forward begins.
                    self.optimizer.zero_grad()
                    meta_val_losses = self._compute_meta_validation_losses(
                        meta_loader,
                        meta_batch_size=weight_args.get('meta_batch_size', None)
                    )
                    self.model.stage_validation_gradient(meta_val_losses, **weight_args)
                    weight_args['val_losses'] = None
                    self._current_meta_val_losses = meta_val_losses.detach().cpu().tolist()
                    meta_gradient_staged = True
                if not self.multi_input:
                    train_inputs, train_gts = self._process_data(train_loader)
                    train_preds = self.model(train_inputs)
                    train_preds = self.process_preds(train_preds)
                    train_losses = self._compute_loss(train_preds, train_gts)
                    self.meter.update(train_preds, train_gts)
                else:
                    train_losses = torch.zeros(self.task_num).to(self.device)
                    for tn, task in enumerate(self.task_name):
                        train_input, train_gt = self._process_data(train_loader[task])
                        train_pred = self.model(train_input, task)
                        train_pred = train_pred[task]
                        train_pred = self.process_preds(train_pred, task)
                        train_losses[tn] = self._compute_loss(train_pred, train_gt, task)
                        self.meter.update(train_pred, train_gt, task)

                diagnostics = None
                if self.task_diagnostics_path is not None and global_step % self.task_diagnostics_every == 0:
                    diagnostics = self._compute_task_diagnostics(train_losses)

                self.optimizer.zero_grad()
                if self.requires_meta_val:
                    if not meta_gradient_staged:
                        meta_val_losses = self._compute_meta_validation_losses(
                            meta_loader,
                            meta_batch_size=weight_args.get('meta_batch_size', None)
                        )
                        weight_args['val_losses'] = meta_val_losses
                        self._current_meta_val_losses = meta_val_losses.detach().cpu().tolist()
                else:
                    self._current_meta_val_losses = [np.nan] * self.task_num
                w = self.model.backward(train_losses, **weight_args)
                if w is not None:
                    self.batch_weight[:, epoch, batch_index] = np.asarray(w, dtype=float)
                self._current_batch_weights = (
                    self.batch_weight[:, epoch, batch_index].tolist()
                )
                dot_product_scores = getattr(self.model, 'latest_dot_product_scores', None)
                if dot_product_scores is None:
                    self._current_dot_product_scores = [np.nan] * self.task_num
                else:
                    self._current_dot_product_scores = dot_product_scores.detach().cpu().tolist()
                scaled_dot_product_scores = getattr(self.model, 'latest_scaled_dot_product_scores', None)
                if scaled_dot_product_scores is None:
                    self._current_scaled_dot_product_scores = [np.nan] * self.task_num
                else:
                    self._current_scaled_dot_product_scores = scaled_dot_product_scores.detach().cpu().tolist()
                score_scale = getattr(self.model, 'latest_score_scale', None)
                if score_scale is None:
                    self._current_score_scale = np.nan
                else:
                    self._current_score_scale = float(score_scale.detach().cpu().item())
                alignment_matrix = getattr(self.model, 'latest_alignment_matrix', None)
                if alignment_matrix is None:
                    self._current_alignment_matrix = [np.nan] * (self.task_num * self.task_num)
                else:
                    self._current_alignment_matrix = alignment_matrix.detach().cpu().reshape(-1).tolist()
                self._log_task_diagnostics(epoch, batch_index, global_step, diagnostics)
                self.optimizer.step()
            
            self.meter.record_time('end')
            self.meter.get_score()
            self.model.train_loss_buffer[:, epoch] = self.meter.loss_item
            self.meter.display(epoch=epoch, mode='train')
            self._log_epoch_metrics('train', epoch)
            self.meter.reinit()
            
            run_eval = ((epoch + 1) % self.test_every == 0) or (epoch == epochs - 1)
            if initial_eval_logged and epoch == 0 and self.test_every == 1 and epochs > 1:
                run_eval = False
            val_improvement = None
            if run_eval:
                # Evaluation needs a separate forward pass.  Releasing gradient buffers and
                # unused allocator blocks here does not affect the completed optimiser step,
                # but avoids carrying the final training batch's memory into validation/test.
                self.optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if val_dataloaders is not None:
                    self.meter.has_val = True
                    val_improvement = self.test(val_dataloaders, epoch, mode='val', return_improvement=True)
                if test_dataloaders is not None:
                    self.test(test_dataloaders, epoch, mode='test')
            if self.scheduler is not None:
                if self.scheduler_param['scheduler'] == 'reduce' and val_dataloaders is not None:
                    if val_improvement is not None:
                        self.scheduler.step(val_improvement)
                else:
                    self.scheduler.step()
            if self.save_best and self.save_path is not None and self.meter.best_result['epoch'] == epoch:
                torch.save(self.model.state_dict(), os.path.join(self.save_path, 'best.pt'))
                print('Save Model {} to {}'.format(epoch, os.path.join(self.save_path, 'best.pt')))
        self.meter.display_best_result()
        if return_weight:
            return self.batch_weight


    def test(self, test_dataloaders, epoch=None, mode='test', return_improvement=False):
        r'''The test process of multi-task learning.

        Args:
            test_dataloaders (dict or torch.utils.data.DataLoader): If ``multi_input`` is ``True``, \
                            it is a dictionary of name-dataloader pairs. Otherwise, it is a single \
                            dataloader which returns data and a dictionary of name-label pairs in each iteration.
            epoch (int, default=None): The current epoch. 
        '''
        test_loader, test_batch = self._prepare_dataloaders(test_dataloaders)
        
        self.model.eval()
        self.meter.record_time('begin')
        with torch.no_grad():
            if not self.multi_input:
                for batch_index in range(test_batch):
                    test_inputs, test_gts = self._process_data(test_loader)
                    test_preds = self.model(test_inputs)
                    test_preds = self.process_preds(test_preds)
                    test_losses = self._compute_loss(test_preds, test_gts)
                    self.meter.update(test_preds, test_gts)
            else:
                for tn, task in enumerate(self.task_name):
                    for batch_index in range(test_batch[tn]):
                        test_input, test_gt = self._process_data(test_loader[task])
                        test_pred = self.model(test_input, task)
                        test_pred = test_pred[task]
                        test_pred = self.process_preds(test_pred)
                        test_loss = self._compute_loss(test_pred, test_gt, task)
                        self.meter.update(test_pred, test_gt, task)
        self.meter.record_time('end')
        self.meter.get_score()
        self.meter.display(epoch=epoch, mode=mode)
        self._log_epoch_metrics(mode, epoch)
        improvement = self.meter.improvement
        self.meter.reinit()
        if return_improvement:
            return improvement
