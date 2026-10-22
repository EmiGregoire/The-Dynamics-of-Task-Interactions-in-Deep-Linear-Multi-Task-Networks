import math
import torch
import torch.nn as nn

from LibMTL.architecture.abstract_arch import AbsArchitecture


class HybridHPS(AbsArchitecture):
    r"""Hybrid Hard Parameter Sharing (HybridHPS).

    This architecture behaves like HPS in the forward pass, but partitions the
    shared encoder into a uniform-weighted group and a dynamically-weighted
    group. The split is block-based for ResNet and SegNet encoders.  For the
    NYUv2 two-layer-ReLU encoder, the split is over output channels of its
    single shared convolution.
    """
    def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs):
        super(HybridHPS, self).__init__(task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs)
        self.encoder = self.encoder_class()

        self.hybridfraction = float(self.kwargs.get('hybridfraction', 0.0))
        self.hybridfraction = min(max(self.hybridfraction, 0.0), 1.0)

        self._shared_block_specs = self._build_shared_block_specs()
        self._uniform_block_names, self._dynamic_block_names = self._split_shared_blocks()

    def _build_shared_block_specs(self):
        if self._is_segnet_mtan_encoder():
            return self._build_segnet_mtan_block_specs()
        if self._is_two_layer_relu_encoder():
            # The actual parameter partition is channel-wise; this one entry
            # simply documents the one shared convolutional module.
            return [('conv1_channels', self.encoder.conv1)]

        return self._build_resnet_like_block_specs()

    def _is_two_layer_relu_encoder(self):
        """Identify the lightweight NYUv2 shared encoder without naming it.

        It has one convolution, followed by a parameter-free ReLU, rather
        than the named ResNet/SegNet blocks handled elsewhere in this class.
        """
        return (
            isinstance(getattr(self.encoder, 'conv1', None), nn.Conv2d)
            and isinstance(getattr(self.encoder, 'relu', None), nn.ReLU)
            and hasattr(self.encoder, 'feature_dim')
            and not any(hasattr(self.encoder, name) for name in ['layer1', 'layer2', 'layer3', 'layer4'])
        )

    def _build_resnet_like_block_specs(self):
        block_specs = []
        stem_modules = []
        if hasattr(self.encoder, 'conv1'):
            stem_modules.append(self.encoder.conv1)
        if hasattr(self.encoder, 'bn1'):
            stem_modules.append(self.encoder.bn1)
        if len(stem_modules) > 0:
            block_specs.append(('stem', nn.ModuleList(stem_modules)))

        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            if hasattr(self.encoder, layer_name):
                block_specs.append((layer_name, getattr(self.encoder, layer_name)))

        if len(block_specs) == 0:
            raise ValueError('HybridHPS currently expects an encoder with named shared blocks such as conv1/bn1/layer1-4')

        return block_specs

    def _is_segnet_mtan_encoder(self):
        required_attrs = [
            'encoder_block',
            'conv_block_enc',
            'decoder_block',
            'conv_block_dec',
            'encoder_att',
            'decoder_att',
            'encoder_block_att',
            'decoder_block_att',
        ]
        return all(hasattr(self.encoder, attr) for attr in required_attrs)

    def _build_segnet_mtan_block_specs(self):
        block_specs = []
        stage_num = len(self.encoder.encoder_block)

        segnet_lists = [
            self.encoder.conv_block_enc,
            self.encoder.decoder_block,
            self.encoder.conv_block_dec,
            self.encoder.encoder_block_att,
            self.encoder.decoder_block_att,
        ]
        if any(len(module_list) != stage_num for module_list in segnet_lists):
            raise ValueError('HybridHPS expected matching SegNet_MTAN stage counts')

        for task_att in self.encoder.encoder_att:
            if len(task_att) != stage_num:
                raise ValueError('HybridHPS expected matching SegNet_MTAN encoder attention stage counts')
        for task_att in self.encoder.decoder_att:
            if len(task_att) != stage_num:
                raise ValueError('HybridHPS expected matching SegNet_MTAN decoder attention stage counts')

        for stage_idx in range(stage_num):
            modules = [
                self.encoder.encoder_block[stage_idx],
                self.encoder.conv_block_enc[stage_idx],
                self.encoder.encoder_block_att[stage_idx],
            ]
            modules.extend(task_att[stage_idx] for task_att in self.encoder.encoder_att)
            block_specs.append(('enc{}'.format(stage_idx), nn.ModuleList(modules)))

        for exec_idx in range(stage_num):
            stage_idx = stage_num - exec_idx - 1
            modules = [
                self.encoder.decoder_block[stage_idx],
                self.encoder.conv_block_dec[stage_idx],
                self.encoder.decoder_block_att[stage_idx],
            ]
            modules.extend(task_att[stage_idx] for task_att in self.encoder.decoder_att)
            block_specs.append(('dec{}'.format(stage_idx), nn.ModuleList(modules)))

        self._validate_segnet_mtan_block_specs(block_specs)
        return block_specs

    def _validate_segnet_mtan_block_specs(self, block_specs):
        all_shared_param_ids = set(id(param) for param in self.encoder.parameters())
        seen_param_ids = set()

        for block_name, module in block_specs:
            for param in module.parameters():
                param_id = id(param)
                if param_id in seen_param_ids:
                    raise ValueError('HybridHPS assigned a SegNet_MTAN parameter to multiple blocks near {}'.format(block_name))
                seen_param_ids.add(param_id)

        missing_param_count = len(all_shared_param_ids - seen_param_ids)
        extra_param_count = len(seen_param_ids - all_shared_param_ids)
        if missing_param_count > 0 or extra_param_count > 0:
            raise ValueError(
                'HybridHPS SegNet_MTAN block split is incomplete '
                '(missing={}, extra={})'.format(missing_param_count, extra_param_count)
            )

    def _split_shared_blocks(self):
        if self._is_two_layer_relu_encoder():
            dynamic_channels = self.get_two_layer_relu_channel_partition()['dynamic_channels']
            uniform_channels = self.get_two_layer_relu_channel_partition()['uniform_channels']
            uniform_names = [] if uniform_channels == 0 else ['conv1_channels_uniform']
            dynamic_names = [] if dynamic_channels == 0 else ['conv1_channels_dynamic']
            return uniform_names, dynamic_names

        block_num = len(self._shared_block_specs)
        dynamic_block_num = int(math.ceil(block_num * self.hybridfraction))

        if self.hybridfraction > 0.0 and dynamic_block_num == 0:
            dynamic_block_num = 1
        dynamic_block_num = min(dynamic_block_num, block_num)

        if dynamic_block_num == 0:
            dynamic_block_names = []
            uniform_block_names = [name for name, _ in self._shared_block_specs]
        else:
            dynamic_block_names = [name for name, _ in self._shared_block_specs[-dynamic_block_num:]]
            uniform_block_names = [name for name, _ in self._shared_block_specs[:-dynamic_block_num]]

        return uniform_block_names, dynamic_block_names

    def get_share_params(self):
        return self.encoder.parameters()

    def zero_grad_share_params(self):
        self.encoder.zero_grad()

    def get_share_blocks(self):
        return [name for name, _ in self._shared_block_specs]

    def get_dynamic_share_block_names(self):
        return list(self._dynamic_block_names)

    def get_uniform_share_block_names(self):
        return list(self._uniform_block_names)

    def get_two_layer_relu_channel_partition(self):
        """Return deterministic masks for the two-layer-ReLU channel split.

        The last ``ceil(hybridfraction * hidden_dim)`` output filters of
        ``conv1`` are dynamic.  The remaining filters are uniformly weighted.
        Biases, when present, use the same output-channel masks.  This method
        is deliberately separate from the block-level parameter accessors:
        tensor slices are not independent PyTorch parameters.
        """
        if not self._is_two_layer_relu_encoder():
            raise ValueError('Channel-wise hybrid partitions are only defined for the two-layer-ReLU encoder')

        conv = self.encoder.conv1
        total_channels = conv.out_channels
        dynamic_channels = min(total_channels, int(math.ceil(total_channels * self.hybridfraction)))
        uniform_channels = total_channels - dynamic_channels

        dynamic_output_mask = torch.zeros(total_channels, dtype=torch.bool, device=conv.weight.device)
        if dynamic_channels > 0:
            dynamic_output_mask[-dynamic_channels:] = True
        uniform_output_mask = ~dynamic_output_mask

        dynamic_weight_mask = dynamic_output_mask[:, None, None, None].expand_as(conv.weight)
        uniform_weight_mask = uniform_output_mask[:, None, None, None].expand_as(conv.weight)
        return {
            'total_channels': total_channels,
            'dynamic_channels': dynamic_channels,
            'uniform_channels': uniform_channels,
            'dynamic_output_mask': dynamic_output_mask,
            'uniform_output_mask': uniform_output_mask,
            'dynamic_parameter_masks': {
                conv.weight: dynamic_weight_mask,
                **({conv.bias: dynamic_output_mask} if conv.bias is not None else {}),
            },
            'uniform_parameter_masks': {
                conv.weight: uniform_weight_mask,
                **({conv.bias: uniform_output_mask} if conv.bias is not None else {}),
            },
        }

    def get_dynamic_share_params(self):
        if self._is_two_layer_relu_encoder():
            raise ValueError(
                'Two-layer-ReLU HybridHPS uses a channel-wise partition; '
                'use a channel-aware hybrid weighting method rather than block-level parameter accessors'
            )
        for name, module in self._shared_block_specs:
            if name in self._dynamic_block_names:
                for param in module.parameters():
                    yield param

    def get_uniform_share_params(self):
        if self._is_two_layer_relu_encoder():
            raise ValueError(
                'Two-layer-ReLU HybridHPS uses a channel-wise partition; '
                'use a channel-aware hybrid weighting method rather than block-level parameter accessors'
            )
        for name, module in self._shared_block_specs:
            if name in self._uniform_block_names:
                for param in module.parameters():
                    yield param

    def zero_grad_dynamic_share_params(self):
        if self._is_two_layer_relu_encoder():
            raise ValueError('Two-layer-ReLU HybridHPS requires channel-wise gradient masking')
        for name, module in self._shared_block_specs:
            if name in self._dynamic_block_names:
                module.zero_grad()

    def zero_grad_uniform_share_params(self):
        if self._is_two_layer_relu_encoder():
            raise ValueError('Two-layer-ReLU HybridHPS requires channel-wise gradient masking')
        for name, module in self._shared_block_specs:
            if name in self._uniform_block_names:
                module.zero_grad()
