# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from collections import OrderedDict
from copy import deepcopy
import numpy as np
import random
from scipy import interpolate

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer, constant_init, trunc_normal_init
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner import BaseModule, ModuleList, _load_checkpoint
from mmcv.utils import to_2tuple
from timm.models.vision_transformer import Mlp

from mmdet.utils import get_root_logger
from mmdet.models.builder import BACKBONES
from mmdet.models.utils.ckpt_convert import swin_converter
from mmdet.models.utils.transformer import PatchEmbed, PatchMerging
from timm.models.vision_transformer import Block


class WidthAttention(nn.Module):

    def __init__(self, dim, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask = None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class WBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.num_heads = num_heads
        self.attn = WidthAttention(dim, num_heads, qkv_bias=True, attn_drop=0, proj_drop=drop)
        self.drop_path1 = build_dropout(dict(type='DropPath', drop_prob=drop_path))

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path2 = build_dropout(dict(type='DropPath', drop_prob=drop_path))

    def init_weight(self):
        print('zero init weight of wblock')
        nn.init.constant_(self.attn.proj.weight, 0.)
        nn.init.constant_(self.attn.proj.bias, 0.)
        nn.init.constant_(self.mlp.fc2.weight, 0.)
        nn.init.constant_(self.mlp.fc2.bias, 0.)

    def forward(self, x, hw_shape):
        H, W = hw_shape
        B, N, C = x.shape
        if B % 6 == 0:
            num_views = 6
        elif B % 5 == 0:
            num_views = 5
        else:
            raise ValueError(f"Invalid batch size {B}")
        B_ = B // num_views
        x_ = x.reshape(B_, num_views, H, W, C)
        x_ = torch.einsum('b n h w c -> b h n w c', x_)
        x_ = x_.reshape(B_*H, num_views*W, C)

        # 6B N C
        x_norm = self.norm1(x_)
        attn_res = self.attn(x_norm)
        attn_res = attn_res.reshape(B_, H, num_views, W, C)
        attn_res = torch.einsum('bhnwc->bnhwc', attn_res).reshape(B, N, C)
        x = x + self.drop_path1(attn_res)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class WindowMSA(BaseModule):
    """Window based multi-head self-attention (W-MSA) module with relative
    position bias.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop_rate (float, optional): Dropout ratio of output. Default: 0.
        init_cfg (dict | None, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 init_cfg=None,
                 use_bias=True):

        super().__init__()
        self.embed_dims = embed_dims
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims**-0.5
        self.init_cfg = init_cfg
        self.use_bias = use_bias

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # About 2x faster than original impl
        Wh, Ww = self.window_size
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer('relative_position_index', rel_position_index)

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.softmax = nn.Softmax(dim=-1)

    def init_weights(self):
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        """
        Args:

            x (tensor): input features with shape of (num_windows*B, N, C)
            mask (tensor | None, Optional): mask with shape of (num_windows,
                Wh*Ww, Wh*Ww), value should be between (-inf, 0].
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        # 3, B, num_heads, N, C
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.use_bias:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1],
                    self.window_size[0] * self.window_size[1],
                    -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            for i in range(len(mask)):
                mask[i] = 1 - mask[i].reshape(B, 1, 1, N)
            assert len(mask) == 2
            mask_new = mask[0] * mask[0].transpose(2, 3) + mask[1] * mask[1].transpose(2, 3)
            mask_new = 1 - mask_new
            if mask_new.dtype == torch.float16:
                attn = attn - 65500 * mask_new
            else:
                attn = attn - 1e30 * mask_new

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


class ShiftWindowMSA(BaseModule):
    """Shifted Window Multihead Self-Attention Module.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window.
        shift_size (int, optional): The shift step of each window towards
            right-bottom. If zero, act as regular window-msa. Defaults to 0.
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v.
            Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Defaults: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Defaults: 0.
        proj_drop_rate (float, optional): Dropout ratio of output.
            Defaults: 0.
        dropout_layer (dict, optional): The dropout_layer used before output.
            Defaults: dict(type='DropPath', drop_prob=0.).
        init_cfg (dict, optional): The extra config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 shift_size=0,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0,
                 proj_drop_rate=0,
                 dropout_layer=dict(type='DropPath', drop_prob=0.),
                 init_cfg=None,
                 use_bias=True):
        super().__init__(init_cfg)

        self.window_size = window_size
        self.shift_size = shift_size
        self.shift_size = 0
        assert 0 <= self.shift_size < self.window_size

        self.w_msa = WindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=to_2tuple(window_size),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            init_cfg=None,
            use_bias=use_bias)

        self.drop = build_dropout(dropout_layer)

    def forward(self, query, hw_shape, mask=None):
        B, L, C = query.shape
        H, W = hw_shape
        assert L == H * W, 'input feature has wrong size'
        query = query.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = query.shape[1], query.shape[2]

        shifted_query = query
        attn_mask = None

        # nW*B, window_size, window_size, C
        query_windows = self.window_partition(shifted_query)
        # nW*B, window_size*window_size, C
        query_windows = query_windows.view(-1, self.window_size**2, C)

        all_masks = None
        if mask is not None:
            all_masks = []
            for _mask in mask:

                _mask = _mask.unsqueeze(0).repeat(B, 1, 1)   # B, H, W
                _mask = torch.nn.functional.interpolate(_mask[:, None], size=(H, W), mode='nearest')
                _mask = _mask.permute(0, 2, 3, 1).contiguous()
                _mask = F.pad(_mask, (0, 0, 0, pad_r, 0, pad_b), value=1.)
                _mask = self.window_partition(_mask)
                _mask = _mask.view(-1, self.window_size * self.window_size, 1)
                all_masks.append(_mask)

        # W-MSA/SW-MSA (nW*B, window_size*window_size, C)
        attn_windows = self.w_msa(query_windows, mask=all_masks)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size,
                                         self.window_size, C)

        # B H' W' C
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)
        x = shifted_x

        if pad_r > 0 or pad_b:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        x = self.drop(x)
        return x

    def window_reverse(self, windows, H, W):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            H (int): Height of image
            W (int): Width of image
        Returns:
            x: (B, H, W, C)
        """
        window_size = self.window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size,
                         window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def window_partition(self, x):
        """
        Args:
            x: (B, H, W, C)
        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(B, H // window_size, window_size, W // window_size,
                   window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows


class SwinBlock(BaseModule):
    """"
    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        window_size (int, optional): The local window scale. Default: 7.
        shift (bool, optional): whether to shift window or not. Default False.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float, optional): Stochastic depth rate. Default: 0.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 window_size=7,
                 shift=False,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None,
                 use_bias=True):

        super(SwinBlock, self).__init__()

        self.init_cfg = init_cfg
        self.with_cp = with_cp

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = ShiftWindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2 if shift else 0,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            init_cfg=None,
            use_bias=use_bias)

        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=2,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg,
            add_identity=True,
            init_cfg=None)

    def forward(self, x, hw_shape, mask=None):

        def _inner_forward(x):
            identity = x
            x = self.norm1(x)
            x = self.attn(x, hw_shape, mask=mask)

            x = x + identity

            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x


class SwinBlockSequence(BaseModule):
    """Implements one stage in Swin Transformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        depth (int): The number of blocks in this stage.
        window_size (int, optional): The local window scale. Default: 7.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float | list[float], optional): Stochastic depth
            rate. Default: 0.
        downsample (BaseModule | None, optional): The downsample operation
            module. Default: None.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 depth,
                 window_size=7,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 downsample=None,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        else:
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]

        self.blocks = ModuleList()
        for i in range(depth):
            use_bias = True
            this_window_size = window_size
            block = SwinBlock(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                window_size=this_window_size,
                shift=False if i % 2 == 0 else True,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None,
                use_bias=use_bias)
            self.blocks.append(block)

        self.downsample = downsample

    def forward(self, x, hw_shape, mask=None):
        for block in self.blocks:
            x = block(x, hw_shape, mask=mask)

        if self.downsample:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            return x_down, down_hw_shape, x, hw_shape
        else:
            return x, hw_shape, x, hw_shape


@BACKBONES.register_module(force=True)
class SwinTransformerBEV(BaseModule):
    """ Swin Transformer
    A PyTorch implement of : `Swin Transformer:
    Hierarchical Vision Transformer using Shifted Windows`  -
        https://arxiv.org/abs/2103.14030

    Inspiration from
    https://github.com/microsoft/Swin-Transformer

    Args:
        pretrain_img_size (int | tuple[int]): The size of input image when
            pretrain. Defaults: 224.
        in_channels (int): The num of input channels.
            Defaults: 3.
        embed_dims (int): The feature dimension. Default: 96.
        patch_size (int | tuple[int]): Patch size. Default: 4.
        window_size (int): Window size. Default: 7.
        mlp_ratio (int): Ratio of mlp hidden dim to embedding dim.
            Default: 4.
        depths (tuple[int]): Depths of each Swin Transformer stage.
            Default: (2, 2, 6, 2).
        num_heads (tuple[int]): Parallel attention heads of each Swin
            Transformer stage. Default: (3, 6, 12, 24).
        strides (tuple[int]): The patch merging or patch embedding stride of
            each Swin Transformer stage. (In swin, we set kernel size equal to
            stride.) Default: (4, 2, 2, 2).
        out_indices (tuple[int]): Output from which stages.
            Default: (0, 1, 2, 3).
        qkv_bias (bool, optional): If True, add a learnable bias to query, key,
            value. Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        patch_norm (bool): If add a norm layer for patch embed and patch
            merging. Default: True.
        drop_rate (float): Dropout rate. Defaults: 0.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Defaults: 0.1.
        use_abs_pos_embed (bool): If True, add absolute position embedding to
            the patch embedding. Defaults: False.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LN').
        norm_cfg (dict): Config dict for normalization layer at
            output of backone. Defaults: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        pretrained (str, optional): model pretrained path. Default: None.
        convert_weights (bool): The flag indicates whether the
            pre-trained model is from the original repo. We may need
            to convert some keys to make it compatible.
            Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            Default: -1 (-1 means not freezing any parameters).
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 pretrain_img_size=224,
                 in_channels=3,
                 embed_dims=128,
                 patch_size=4,
                 window_size=(16, 16, 16, 8),
                 mlp_ratio=4,
                 depths=(2, 2, 18, 2),
                 num_heads=(4, 8, 16, 32),
                 strides=(4, 2, 2, 2),
                 out_indices=(2, 3),
                 qkv_bias=True,
                 qk_scale=None,
                 patch_norm=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.0,
                 use_abs_pos_embed=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 pretrained=None,
                 convert_weights=False,
                 frozen_stages=-1,
                 init_cfg=None,
                 # decoder
                 decoder_dim=512,
                 decoder_num_heads=16,
                 decoder_depth=8,):
        self.convert_weights = convert_weights
        self.frozen_stages = frozen_stages
        if isinstance(pretrain_img_size, int):
            pretrain_img_size = to_2tuple(pretrain_img_size)
        elif isinstance(pretrain_img_size, tuple):
            if len(pretrain_img_size) == 1:
                pretrain_img_size = to_2tuple(pretrain_img_size[0])
            assert len(pretrain_img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(pretrain_img_size)}'

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            self.init_cfg = init_cfg
        else:
            raise TypeError('pretrained must be a str or None')

        super(SwinTransformerBEV, self).__init__(init_cfg=init_cfg)

        num_layers = len(depths)
        self.out_indices = out_indices
        self.use_abs_pos_embed = use_abs_pos_embed

        assert strides[0] == patch_size, 'Use non-overlapping patch embed.'

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=strides[0],
            norm_cfg=norm_cfg if patch_norm else None,
            init_cfg=None)

        if self.use_abs_pos_embed:
            patch_row = pretrain_img_size[0] // patch_size
            patch_col = pretrain_img_size[1] // patch_size
            num_patches = patch_row * patch_col
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros((1, embed_dims, patch_row, patch_col)))

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # set stochastic depth decay rule
        total_depth = sum(depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]

        self.stages = ModuleList()
        in_channels = embed_dims
        for i in range(num_layers):
            if i < num_layers - 1:
                downsample = PatchMerging(
                    in_channels=in_channels,
                    out_channels=2 * in_channels,
                    stride=strides[i + 1],
                    norm_cfg=norm_cfg if patch_norm else None,
                    init_cfg=None)
            else:
                downsample = None

            stage = SwinBlockSequence(
                embed_dims=in_channels,
                num_heads=num_heads[i],
                feedforward_channels=mlp_ratio * in_channels,
                depth=depths[i],
                window_size=window_size[i],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=downsample,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp if isinstance(with_cp, bool) else with_cp > i,
                init_cfg=None)
            self.stages.append(stage)
            if downsample:
                in_channels = downsample.out_channels

        self.num_features = [int(embed_dims * 2**i) for i in range(num_layers)]
        # Add a norm layer for each output
        for i in out_indices:
            layer = build_norm_layer(norm_cfg, self.num_features[i])[1]
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)

        # add for mim
        self.encoder_stride = 32
        self.decoder_dim = decoder_dim
        self.num_batch_imgs = 2

        # upsample one stage
        self.up_p4_fc = nn.Linear(self.num_features[3], self.num_features[2])
        self.up_smooth_p4 = SwinBlock(
            self.num_features[2], num_heads[2], mlp_ratio * self.num_features[2])
        self.final_norm = nn.LayerNorm(self.num_features[2])

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_dim))
        trunc_normal_(self.mask_token, mean=0., std=.02)

        patch_row = pretrain_img_size[0] // self.encoder_stride
        patch_col = pretrain_img_size[1] // self.encoder_stride
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.decoder_dim, \
            patch_row, patch_col), requires_grad=False)
        self.decoder_embed_new = nn.Linear(self.num_features[2], self.decoder_dim)
        self.decoder_blocks = nn.ModuleList([
            Block(self.decoder_dim, decoder_num_heads, mlp_ratio, qkv_bias=True)
            for i in range(decoder_depth)])
        self.decoder_norm = nn.LayerNorm(self.decoder_dim)

        self.decoder_blocks_pix = nn.ModuleList([
            Block(self.decoder_dim, decoder_num_heads, mlp_ratio, qkv_bias=True)
            for i in range(decoder_depth // 2)])
        self.decoder_norm_pix = nn.LayerNorm(self.decoder_dim)

        self.cam_pose = PoseSEModule(self.decoder_dim)
        self.decoder_cross_module = nn.ModuleList()
        for i in range(3):
            m = WBlock(self.decoder_dim, decoder_num_heads)
            self.decoder_cross_module.append(m)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformerBEV, self).train(mode)
        # self._freeze_stages()

    def _freeze_stages(self):
        # as pretrain use cosine
        # self.absolute_pos_embed.requires_grad = False
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            if self.use_abs_pos_embed:
                self.absolute_pos_embed.requires_grad = False
            self.drop_after_pos.eval()

        for i in range(1, self.frozen_stages + 1):

            if (i - 1) in self.out_indices:
                norm_layer = getattr(self, f'norm{i-1}')
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False

            m = self.stages[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        logger = get_root_logger()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            # if self.use_abs_pos_embed:
                # trunc_normal_(self.absolute_pos_embed, std=0.02)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, 1.0)
                if hasattr(m, 'init_weight'):
                    m.init_weight()
        else:
            for m in self.modules():
                if hasattr(m, 'init_weight'):
                    m.init_weight()
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = _load_checkpoint(
                self.init_cfg['checkpoint'], logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt
            if self.convert_weights:
                # supported loading weight from original repo,
                _state_dict = swin_converter(_state_dict)

            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('backbone.'):
                    if 'relative_position_index' in k:
                        continue
                    state_dict[k[9:]] = v

            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            # reshape absolute position embedding
            if state_dict.get('absolute_pos_embed') is not None:
                absolute_pos_embed = state_dict['absolute_pos_embed']
                N1, L, C1 = absolute_pos_embed.size()
                N2, C2, H, W = self.absolute_pos_embed.size()
                if N1 != N2 or C1 != C2 or L != H * W:
                    logger.warning('Error in loading absolute_pos_embed, pass')
                else:
                    state_dict['absolute_pos_embed'] = absolute_pos_embed.view(
                        N2, H, W, C2).permute(0, 3, 1, 2).contiguous()

            if state_dict.get('decoder_pos_embed') is not None:
                decoder_pos_embed = state_dict['decoder_pos_embed']
                N1, L, C1 = decoder_pos_embed.size()
                N2, C2, H, W = self.decoder_pos_embed.size()
                if N1 != N2 or C1 != C2 or L != H * W:
                    logger.warning('Error in loading absolute_pos_embed, pass')
                else:
                    state_dict['decoder_pos_embed'] = decoder_pos_embed.view(
                        N2, H, W, C2).permute(0, 3, 1, 2).contiguous()

            # interpolate position bias table if needed
            relative_position_bias_table_keys = [
                k for k in state_dict.keys()
                if 'relative_position_bias_table' in k
            ]
            for table_key in relative_position_bias_table_keys:
                table_pretrained = state_dict[table_key]
                table_current = self.state_dict()[table_key]
                L1, nH1 = table_pretrained.size()
                L2, nH2 = table_current.size()
                if nH1 != nH2:
                    logger.warning(f'Error in loading {table_key}, pass')
                elif L1 != L2:
                    S1 = int(L1**0.5)
                    S2 = int(L2**0.5)
                    def geometric_progression(a, r, n):
                        return a * (1.0 - r ** n) / (1.0 - r)

                    left, right = 1.01, 1.5
                    while right - left > 1e-6:
                        q = (left + right) / 2.0
                        gp = geometric_progression(1, q, S1 // 2)
                        if gp > S2 // 2:
                            right = q
                        else:
                            left = q
                    dis = []
                    cur = 1
                    for i in range(S1 // 2):
                        dis.append(cur)
                        cur += q ** (i + 1)

                    r_ids = [-_ for _ in reversed(dis)]

                    x = r_ids + [0] + dis
                    y = r_ids + [0] + dis

                    t = S2 // 2.0
                    dx = np.arange(-t, t + 0.1, 1.0)
                    dy = np.arange(-t, t + 0.1, 1.0)

                    # print("Original positions = %s" % str(x))
                    # print("Target positions = %s" % str(dx))
                    all_rel_pos_bias = []

                    for i in range(nH2):
                        z = table_pretrained[:, i].view(S1, S1).float().numpy()
                        f = interpolate.interp2d(x, y, z, kind='cubic')
                        all_rel_pos_bias.append(
                            torch.Tensor(f(dx, dy)).contiguous().view(-1, 1).to(table_pretrained.device))

                    rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)
                    state_dict[table_key] = rel_pos_bias
            
            # load state_dict
            msg = self.load_state_dict(state_dict, False)
            logger.info(msg)

    def random_masking(self, x, mask_ratio=0.5):
        B, C, H, W = x.shape
        out_H = H // self.encoder_stride
        out_W = W // self.encoder_stride
        seq_l = out_H * out_W

        noise = torch.rand(1, 1, seq_l, device=x.device)  # noise in [0, 1]
        noise_idx = torch.argsort(noise, dim=2)
        # ascend: small is keep, large is remove
        num_mask_tokens = int(seq_l * mask_ratio)

        masks = []
        mask_idxs = []
        nonmask_idxs = []
        for i in range(self.num_batch_imgs):
            start = num_mask_tokens * i
            end = num_mask_tokens * (i + 1)
            mask_idx1 = noise_idx[:, :, :start]
            mask_idx2 = noise_idx[:, :, end:]
            non_mask_idx = noise_idx[:, :, start:end]
            mask_idx = torch.cat([mask_idx1, mask_idx2], dim=2)
            mask = torch.zeros([1, 1, seq_l], device=x.device)

            mask.scatter_(2, mask_idx, 1)
            mask = mask.reshape(out_H, out_W).contiguous()

            masks.append(mask)
            mask_idxs.append(mask_idx.squeeze())
            nonmask_idxs.append(non_mask_idx.squeeze())

        return masks, mask_idxs, nonmask_idxs

    def scale_mask(self, mask, out_size, to_3d=True):
        mask = torch.nn.functional.interpolate(mask, size=out_size, mode='nearest')
        if to_3d:
            return mask.flatten(2, 3).transpose(1, 2)
        return mask

    def _scale_add(self, x, y, input_hw, output_hw):
        B, N, C = x.shape
        x = x.reshape(B, *input_hw, C).permute(0, 3, 1, 2)
        x = F.interpolate(x, size=output_hw, mode='nearest')
        x = x.reshape(B, C, -1).permute(0, 2, 1)
        return x + y

    def forward(
            self, x,
            sensor2ego, cam_intrinsic, img_aug_matrix
        ):
        masks, mask_idxs, nonmask_idxs = self.random_masking(x)

        x, hw_shape = self.patch_embed(x)

        if self.use_abs_pos_embed:
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=hw_shape, mode='bicubic')
            x = x + absolute_pos_embed.flatten(2).transpose(1, 2)
        x = self.drop_after_pos(x)

        outs = []
        all_hw_shapes = []
        for i, stage in enumerate(self.stages):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape, mask=masks)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(out)
                outs.append(out)
            all_hw_shapes.append(out_hw_shape)

        # outs: p4, p5
        c4, c5 = outs
        p5 = c5
        p4 = self._scale_add(self.up_p4_fc(p5), c4, all_hw_shapes[-1], all_hw_shapes[-2])
        p4 = self.up_smooth_p4(p4, all_hw_shapes[-2], mask=masks)
        x = self.final_norm(p4)

        x = self.decoder_embed_new(x)

        # add mask
        B, L, C = x.shape
        mask_tokens = self.mask_token.expand(B, L, -1)
        all_mask = []
        for item in masks:
            _mask = self.scale_mask(item[None, None], all_hw_shapes[-2])
            all_mask.append(_mask)

        x1 = x * (1. - all_mask[0]) + mask_tokens * all_mask[0]
        x2 = x * (1. - all_mask[1]) + mask_tokens * all_mask[1]
        x = torch.cat([x1, x2], dim=0)

        decoder_pos_embed = F.interpolate(self.decoder_pos_embed, size=all_hw_shapes[-2], mode='bicubic')
        x = x + decoder_pos_embed.flatten(2).transpose(1, 2)

        x_pix = None     # for depth reconstruction
        for idx, blk in enumerate(self.decoder_blocks):
            x = blk(x)
            # Cross-view attention
            if idx in [1, 5]:
                x = self.decoder_cross_module[idx // 4](x, all_hw_shapes[-2])
            if idx == 3:
                x_pix = x
        x = self.decoder_norm(x)
        x = x.view(-1, *all_hw_shapes[-2], self.decoder_dim).permute(0, 3, 1, 2).contiguous()

        rots = sensor2ego[..., :3, :3] # 2, 6, 4, 4
        trans = sensor2ego[..., :3, 3]
        intrins = cam_intrinsic[..., :3, :3] # 2, 6, 4, 4
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3] # 2, 6, 4, 4
        mlp_input = self.cam_pose.get_mlp_input(
            rots, trans, intrins, post_rots, post_trans
        )
        mlp_input = [mlp_input for _ in range(self.num_batch_imgs)]
        mlp_input = torch.cat(mlp_input, dim=0)

        x_pix = self.cam_pose(x_pix, mlp_input)

        for idx, blk in enumerate(self.decoder_blocks_pix):
            x_pix = blk(x_pix)
            if idx == 1:
                x_pix = self.decoder_cross_module[2](x_pix, all_hw_shapes[-2])
        x_pix = self.decoder_norm_pix(x_pix)
        x_pix = x_pix.view(-1, *all_hw_shapes[-2], self.decoder_dim).permute(0, 3, 1, 2).contiguous()

        return x, x_pix

class SELayer(nn.Module):

    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Linear(channels, channels, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Linear(channels, channels, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class PoseSEModule(nn.Module):
    def __init__(self, mid_channels=512):
        super().__init__()
        self.bn = nn.BatchNorm1d(22)
        self.depth_mlp = Mlp(22, mid_channels, mid_channels)
        self.depth_se = SELayer(mid_channels)  # NOTE: add camera-aware

    def forward(self, x, mlp_input):
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        depth_se = self.depth_mlp(mlp_input)[:, None, :]
        depth = self.depth_se(x, depth_se)
        return depth

    def get_mlp_input(self, rot, tran, intrin, post_rot, post_tran):
        B, N, _, _ = rot.shape
        mlp_input = torch.stack([
            intrin[:, :, 0, 0],
            intrin[:, :, 1, 1],
            intrin[:, :, 0, 2],
            intrin[:, :, 1, 2],
            post_rot[:, :, 0, 0],
            post_rot[:, :, 0, 1],
            post_tran[:, :, 0],
            post_rot[:, :, 1, 0],
            post_rot[:, :, 1, 1],
            post_tran[:, :, 1],
        ],
                                dim=-1)
        sensor2ego = torch.cat([rot, tran.reshape(B, N, 3, 1)],
                               dim=-1).reshape(B, N, -1)
        mlp_input = torch.cat([mlp_input, sensor2ego], dim=-1)
        return mlp_input
