# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
import timm
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.feature_extraction import create_feature_extractor
from typing import Dict, List

from utils.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias
    

class BackboneConvNeXt(nn.Module):
    """
    ConvNeXt-Small（ImageNet-1K 预训练）导出 stride 8/16/32 三层特征
    """
    def __init__(self,
                 name: str = "convnext_small",
                 train_backbone: bool = True,
                 return_interm_layers: bool = True,
                 convnext_pretrained_path: str = "/home/zjhzjh/workdir/Modify_MonoDGP/convnext_small_22k_1k_224.pth"):
        super().__init__()

        # 22k权重
        # ckpt = torch.load(convnext_pretrained_path, map_location='cpu', weights_only=False)
        # backbone = timm.create_model('convnext_small', pretrained=False)
        # state = {k.replace('model.', ''): v for k, v in ckpt['model'].items() if not k.startswith('head.')}        # 去掉分类头
        # backbone.load_state_dict(state, strict=False)

        # for n, p in backbone.named_parameters():
        #     if not train_backbone or n.startswith(('features.0', 'features.1')):
        #         p.requires_grad_(False)

        # if return_interm_layers:
        #     return_layers = {"stages.1":"0", "stages.2":"1", "stages.3":"2"}
        #     self.strides = [8, 16, 32]
        #     self.num_channels = [192, 384, 768]
        # else:
        #     return_layers = {"stages.3":"0"}
        #     self.strides = [32]
        #     self.num_channels = [768]

        # ------- ① 直接用 torchvision 自带的 1K 权重 ----------
        weights = (
            torchvision.models.ConvNeXt_Small_Weights.IMAGENET1K_V1
            if is_main_process() else None          # 多卡只让 rank-0 下载
        )
        backbone = getattr(torchvision.models, name)(weights=weights)

        # ------- ② 冻结早期层 ----------
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'features.3' not in name and 'features.5' not in name and 'features.7' not in name:
                parameter.requires_grad_(False)

        # ------- ③ 选出要返回的节点 ----------
        if return_interm_layers:
            return_layers = {
                "features.3": "0",   # stride 8   192C
                "features.5": "1",   # stride 16  384C
                "features.7": "2",   # stride 32  768C
            }
            self.strides      = [8, 16, 32]
            self.num_channels = [192, 384, 768]
            # self.num_channels = [256, 512, 1024]
        else:
            return_layers     = {"features.7": "0"}
            self.strides      = [32]
            self.num_channels = [768]

        # ------- ④ 构建特征提取器 ----------
        self.body = create_feature_extractor(backbone, return_nodes=return_layers)

    def forward(self, images):
        xs = self.body(images)
        out = {}
        for name, x in xs.items():
            m = torch.zeros(x.shape[0], x.shape[2], x.shape[3]).to(torch.bool).to(x.device)
            out[name] = NestedTensor(x, m)
        return out


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, images):
        xs = self.body(images)
        out = {}
        for name, x in xs.items():
            m = torch.zeros(x.shape[0], x.shape[2], x.shape[3]).to(torch.bool).to(x.device)
            out[name] = NestedTensor(x, m)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        norm_layer = FrozenBatchNorm2d
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=norm_layer)
        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        super().__init__(backbone, train_backbone, return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, images):
        xs = self[0](images)
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(cfg):
    
    position_embedding = build_position_encoding(cfg)
    return_interm_layers = cfg['masks'] or cfg['num_feature_levels'] > 1
    # backbone = Backbone(cfg['backbone'], cfg['train_backbone'], return_interm_layers, cfg['dilation'])
    backbone = BackboneConvNeXt(cfg['backbone'], cfg['train_backbone'], return_interm_layers)
    model = Joiner(backbone, position_embedding)
    return model