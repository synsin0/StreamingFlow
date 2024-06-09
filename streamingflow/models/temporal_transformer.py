# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence

from mmdet.models.utils.builder import TRANSFORMER
from torch.nn.init import normal_
from mmcv.runner.base_module import BaseModule
from torchvision.transforms.functional import rotate
from streamingflow.layers.transformer_layers.temporal_self_attention import TemporalSelfAttention
from streamingflow.layers.transformer_layers.spatial_cross_attention import MSDeformableAttention3D
from streamingflow.layers.transformer_layers.decoder import CustomMSDeformableAttention
from mmcv.runner import force_fp32, auto_fp16
import math

def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)
    return posemb


@TRANSFORMER.register_module()
class TemporalPerceptionTransformer(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_seqs (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_seqs=3,
                 encoder=None,
                 decoder=None,
                 embed_dims=256,
                 **kwargs):
        super(TemporalPerceptionTransformer, self).__init__(**kwargs)
        self.encoder = build_transformer_layer_sequence(encoder)
        if decoder is not None:
            self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = embed_dims
        self.num_seqs = num_seqs
        
        self.fp16_enabled = False

        self.init_layers()
        self.pos_encoder = nn.Sequential(
            nn.Linear(6, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )
        self.time_encoder = nn.Sequential(
            nn.Linear(1, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )
        self.position_encoder = nn.Sequential(
                nn.Conv2d(4*num_seqs, self.embed_dims*4, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims*4, self.embed_dims, kernel_size=1, stride=1, padding=0),
            )

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.range_embeds = nn.Parameter(torch.Tensor(
            self.num_seqs, self.embed_dims))
        self.voxels_embeds = nn.Parameter(torch.Tensor(
            self.num_seqs, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_seqs, self.embed_dims))
        self.reference_points = nn.Linear(self.embed_dims, 3)

    

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.range_embeds)
        normal_(self.cams_embeds)
        normal_(self.voxels_embeds)

        xavier_init(self.reference_points, distribution='uniform', bias=0.)

    def position_embeding(self, bev_feats, masks=None):
        eps = 1e-5
       
        B, N, C, H, W = bev_feats.shape
        coords_h = torch.arange(H, device=bev_feats[0].device).float()
        coords_w = torch.arange(W, device=bev_feats[0].device).float()
        coords_d = torch.arange(N, device=bev_feats[0].device).float()

        D = coords_d.shape[0]
        coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d])).permute(1, 2, 3, 0) # W, H, D, 3
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)

        coords = coords.view(1, 1, H, W, D, 4,).repeat(B, N, 1, 1, 1, 1)
        coords = coords.permute(0,1,4,5,2,3).view(B*N,-1,H,W)

        coords_position_embeding = self.position_encoder(coords)
        
        return coords_position_embeding.view(B, N, self.embed_dims, H, W)

    @auto_fp16(apply_to=('msmm_feats', 'bev_queries', 'bev_pos'))
    def get_temporal_bev_feature(
            self,
            msmm_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=[0.5, 0.5],
            bev_pos=None,
            bev_timestamps = None,
            ):
        """
        obtain bev features.
        """

  
        bs = msmm_feats['cam_bev_feats'].shape[0]
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        if bev_pos is not None:
            bev_pos = self.pos_encoder(bev_pos)
            bev_pos = bev_pos.permute(1,0,2)
        if bev_timestamps is not None:    
            bev_timestamps = self.time_encoder(bev_timestamps)

        feat_flatten = []
        pos_embeds = []
        spatial_shapes = []
        import ipdb
        ipdb.set_trace()
        for key, value in msmm_feats.items():
            bs, seq, c, h, w  = msmm_feats['cam_bev_feats'].shape
            spatial_shape = (h, w, 3 )
            if key in ['cam_bev_feats']:
                if msmm_feats[key] is not None:
                    feat = msmm_feats[key]
                    pos_embed = self.position_embeding(feat)
                    feat = feat.flatten(3).permute(1, 0, 3, 2)
                    
                    feat = feat + self.cams_embeds[:, None, None, :].to(msmm_feats[key].dtype)  # relative positional encoding
                    feat = feat + bev_pos[:, :, None, :].repeat(1,1,bev_h*bev_w,1).to(msmm_feats[key].dtype) # absolute positional encoding
            if key in ['range_bev_feats']:
                if msmm_feats[key] is not None:
                    feat = msmm_feats[key]
                    pos_embed = self.position_embeding(feat)
                    feat = feat.flatten(3).permute(1, 0, 3, 2)
                    feat = feat + self.range_embeds[:, None, None, :].to(msmm_feats[key].dtype)  # relative positional encoding
                    feat = feat + bev_pos[:, :, None, :].repeat(1,1,bev_h*bev_w,1).to(msmm_feats[key].dtype) # absolute positional encoding
            if key in ['voxel_bev_feats']:
                if msmm_feats[key] is not None:
                    feat = msmm_feats[key]
                    pos_embed = self.position_embeding(feat)
                    feat = feat.flatten(3).permute(1, 0, 3, 2)
                    feat = feat + self.voxels_embeds[:, None, None, :].to(msmm_feats[key].dtype)  # relative positional encoding
                    feat = feat + bev_pos[:, :, None, :].repeat(1,1,bev_h*bev_w,1).to(msmm_feats[key].dtype) # absolute positional encoding

            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)  # List:[Seq, Bs, H*W, C]
            pos_embeds.append(pos_embed)
      
        # feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=bev_pos.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        # feat_flatten = feat_flatten.permute(
        #     0, 2, 1, 3)  # (num_feature_level, H*W, bs, embed_dims)

        bev_embed = self.encoder(
            bev_queries,
            feat_flatten,
            feat_flatten,
            bev_h=bev_h,
            bev_w=bev_w,
            pos_embed = pos_embeds,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )

        return bev_embed

    @auto_fp16(apply_to=('msmm_feats', 'bev_queries', 'object_query_embed', 'prev_bev', 'bev_pos'))
    def forward(self,
                msmm_feats,
                bev_queries,
                object_query_embed,
                bev_h,
                bev_w,
                grid_length=[0.512, 0.512],
                bev_pos=None,
                bev_timestamps = None,
                reg_branches=None,
                cls_branches=None,
                prev_bev=None,
                **kwargs):
        """Forward function for `Detr3DTransformer`.
        Args:
            msmm_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, num_cams, embed_dims, h, w].
            bev_queries (Tensor): (bev_h*bev_w, c)
            bev_pos (Tensor): (bs, embed_dims, bev_h, bev_w)
            object_query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - bev_embed: BEV features
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """

        bev_embed = self.get_temporal_bev_feature(
            msmm_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            **kwargs)  # bev_embed shape: bs, bev_h*bev_w, embed_dims

        bs = msmm_feats[0].size(0)
        query_pos, query = torch.split(
            object_query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        bev_embed = bev_embed.permute(1, 0, 2)

        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
            **kwargs)

        inter_references_out = inter_references

        return bev_embed, inter_states, init_reference_out, inter_references_out
