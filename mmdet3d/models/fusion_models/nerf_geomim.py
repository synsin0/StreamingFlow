from typing import Any, Dict

import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F

from mmdet3d.models.builder import (
    build_backbone,
    build_vtransform,
)
from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models import FUSIONMODELS

from mmdet.models.builder import build_loss
from mmcv.cnn.bricks.conv_module import ConvModule
from .base import Base3DFusionModel
import cv2
import time
from mmdet3d.models import builder

import numpy as np
# from ..model_utils.render_ray import render_rays
# from ..model_utils.nerf_mlp import VanillaNeRFRadianceField
# from ..model_utils.projection import Projector
# from ..model_utils.save_rendered_img import save_rendered_img

# from ..model_utils.nerfstudio_losses import ScaleAndShiftInvariantLoss

__all__ = ["NeRFGeoMIM"]

# occ3d-nuscenes
nusc_class_frequencies = np.array([1163161, 2309034, 188743, 2997643, 20317180, 852476, 243808, 2457947, 
            497017, 2731022, 7224789, 214411435, 5565043, 63191967, 76098082, 128860031, 
            141625221, 2307405309])

def patchify(imgs, encoder_stride):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    p = encoder_stride
    n, c, h, w = imgs.shape
    assert h % p == 0 and w % p == 0

    h1, w1 = h // p, w // p
    x = imgs.reshape(shape=(imgs.shape[0], c, h1, p, w1, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h1 * w1, p**2 * c))
    return x

def unpatchify(x, encoder_stride):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """
    p = encoder_stride
    h = w = int(x.shape[1]**.5)
    assert h * w == x.shape[1]
    c = int(x.shape[2] / p / p)

    x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
    return imgs


@FUSIONMODELS.register_module()
class NeRFGeoMIM(Base3DFusionModel):
    def __init__(
        self,
        encoders: Dict[str, Any],
        bev_feat_dim=256,
        out_dim = 32,
        num_classes = 18,

        # aabb=([-54, -54, -1], [54, 54, 5.4]),
        # near_far_range=[0.2, 50.0],
        # N_samples=40,
        # N_rand=8192,
        # use_nerf_mask=False,
        # nerf_sample_view=6,
        # nerf_mode="occ_volume",
        # squeeze_scale=4,
        # rgb_supervision=True,
        # nerf_density = False,
        # render_testing=False,
        # nvs_depth_supervise = True,
        nerf_head=None,
        test_threshold=8.5,
        use_lss_depth_loss=True,
        use_3d_loss=False,
        balance_cls_weight=True,
        final_softplus=False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.depth_loss = 0.0
        self.fuser = None
        self.decoder = None
        self.heads = None
        # self.use_nvs_loss = True
        self.encoders = nn.ModuleDict()
        if encoders.get("camera") is not None:
            self.encoders["camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["camera"]["backbone"]),
                    "vtransform": build_vtransform(encoders["camera"]["vtransform"]),
                }
            )
        # if encoders.get("lidar") is not None:
        #     if encoders["lidar"]["voxelize"].get("max_num_points", -1) > 0:
        #         voxelize_module = Voxelization(**encoders["lidar"]["voxelize"])
        #     else:
        #         voxelize_module = DynamicScatter(**encoders["lidar"]["voxelize"])
        #     self.encoders["lidar"] = nn.ModuleDict(
        #         {
        #             "voxelize": voxelize_module,
        #             "backbone": build_backbone(encoders["lidar"]["backbone"]),
        #         }
        #     )
        #     self.voxelize_reduce = encoders["lidar"].get("voxelize_reduce", True)

        # for p in self.encoders['lidar'].parameters():
        #     p.requires_grad = False

        # self.prediction_head = nn.Conv3d(
        #     encoders["camera"]["vtransform"]['out_channels'],
        #     bev_feat_dim,
        #     kernel_size=1, stride=1
        # )
        # self.nvs_depth_supervise = nvs_depth_supervise
        # self.nvs_depth_weight = 0.05
        # self.use_nerf_mask = use_nerf_mask
        # self.aabb=aabb
        # self.near_far_range=near_far_range
        # self.N_samples=N_samples
        # self.N_rand=N_rand
        # self.projector = Projector()
        # nerf_feature_dim = encoders["camera"]["vtransform"]['out_channels'] // squeeze_scale
        # self.nerf_mlp = VanillaNeRFRadianceField(
        #     net_depth=4,  # The depth of the MLP.
        #     net_width=256,  # The width of the MLP.
        #     skip_layer=3,  # The layer to add skip layers to.
        #     feature_dim=nerf_feature_dim, # + RGB original img
        #     net_depth_condition=1,  # The depth of the second part of MLP.
        #     net_width_condition=128
        #     )
        # self.nerf_mode = nerf_mode
        # self.nerf_density = nerf_density
        # self.nerf_sample_view = nerf_sample_view


        # self.mean_mapping = nn.Sequential(
        #     nn.Conv3d(
        #          encoders["camera"]["vtransform"]['out_channels'], nerf_feature_dim, kernel_size=1
        #     )
        # )

        # self.render_testing = render_testing

        # if self.nvs_depth_supervise:
        #     self.nvs_depth_loss_func = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1)
        self.out_dim = out_dim
        self.use_3d_loss = use_3d_loss
        self.test_threshold = test_threshold
        self.use_lss_depth_loss = use_lss_depth_loss
        self.balance_cls_weight = balance_cls_weight
        self.final_softplus = final_softplus

        if self.balance_cls_weight:
            self.class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:17] + 0.001)).float()
            self.semantic_loss = nn.CrossEntropyLoss(
                    weight=self.class_weights, reduction="mean"
                )
        else:
            self.semantic_loss = nn.CrossEntropyLoss(reduction="mean")

        self.final_conv = ConvModule(
                        encoders["camera"]["vtransform"]['out_channels'],
                        self.out_dim,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                        conv_cfg=dict(type='Conv3d'))

        if self.final_softplus:
            self.density_mlp = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim*2),
                nn.Softplus(),
                nn.Linear(self.out_dim*2, 2),
                nn.Softplus(),
            )
        else:
            self.density_mlp = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim*2),
                nn.Softplus(),
                nn.Linear(self.out_dim*2, 2),
            )

        self.semantic_mlp = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim*2),
            nn.Softplus(),
            nn.Linear(self.out_dim*2, num_classes-1),
        )

        self.nerf_head = builder.build_head(nerf_head)
      

        self.init_weights()

    def init_weights(self) -> None:
        if "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()

    def extract_camera_features(
        self,
        x,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)
        

        x, x_depth = self.encoders["camera"]["backbone"](
            x, camera2ego, camera_intrinsics, img_aug_matrix  # torch.Size([12, 3, 256, 704]) torch.Size([2, 6, 4, 4])
        )
        # x: torch.Size([24, 512, 16, 44])
        # x_depth: torch.Size([24, 512, 16, 44])
        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()

        num_batch_imgs = int(BN / (B * N))  # 2
        
        outputs = []
        gt_depthes = []
        for i in range(num_batch_imgs):
            this_x = x[B * N * i : B * N * (i + 1), ...]
            this_x = this_x.view(B, N, C, H, W)

            this_depth = x_depth[B * N * i : B * N * (i + 1), ...]
            this_depth = this_depth.view(B, N, C, H, W)  # torch.Size([2, 6, 512, 16, 44])

            this_x, gt_depth = self.encoders["camera"]["vtransform"](
                this_x,
                this_depth,
                points,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                img_metas,
            ) # torch.Size([2, 80, 360, 360, 15])
            # outputs.append(self.prediction_head(this_x))  # torch.Size([2, 256, 180, 180])
            outputs.append(this_x)
            gt_depthes.append(gt_depth)
            if self.encoders["camera"]["vtransform"].ret_depth:
                self.depth_loss = self.depth_loss + self.encoders["camera"]["vtransform"].depth_loss / num_batch_imgs

        return outputs, gt_depthes

    @torch.no_grad()
    @force_fp32()
    def extract_lidar_features(self, x):
        self.encoders["lidar"]['voxelize'].eval()
        self.encoders["lidar"]['backbone'].eval()

        feats, coords, sizes = self.voxelize(x)
        batch_size = coords[-1, 0] + 1
        x = self.encoders["lidar"]["backbone"](feats, coords, batch_size, sizes=sizes)
        return x
    
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.encoders["lidar"]["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes

    @auto_fp16(apply_to=("img"))
    def forward(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas=None,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        if isinstance(img, list):
            raise NotImplementedError
        else:
            outputs = self.forward_single(
                img,
                points,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                gt_masks_bev,
                gt_bboxes_3d,
                gt_labels_3d,
                **kwargs,
            )
            return outputs

    @auto_fp16(apply_to=("img"))
    def forward_single(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        
        rays_info=kwargs['rays_info']

        features = []
        for sensor in (
            self.encoders if self.training else list(self.encoders.keys())[::-1]
        ):
            if sensor == "camera":
                cam_feature, gt_depthes = self.extract_camera_features(
                    img,
                    points,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    metas,
                )
            # elif sensor == "lidar":
            #     with torch.no_grad():
            #         lidar_feature = self.extract_lidar_features(points)
            #         # normalize within grid
            #         lidar_feature = patchify(lidar_feature, 18)
            #         mean = lidar_feature.mean(dim=-1, keepdim=True)
            #         var = lidar_feature.var(dim=-1, keepdim=True)
            #         lidar_feature = (lidar_feature - mean) / (var + 1.e-6)**.5
            #         lidar_feature = unpatchify(lidar_feature, 18)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")
        

        losses_rendering = 0.0

        # compute loss
        losses = dict()

        for mae_idx, cam_feat in enumerate(cam_feature):
            voxel_feats = self.final_conv(cam_feat).permute(0, 4, 3, 2, 1) # bncdhw->bnwhdc

            # predict SDF
            density_prob = self.density_mlp(voxel_feats)
            density = density_prob[..., 0]
            semantic = self.semantic_mlp(voxel_feats)



            # if self.use_3d_loss:      # 3D loss
            #     voxel_semantics = kwargs['voxel_semantics']
            #     mask_camera = kwargs['mask_camera']
            #     assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
            #     loss_occ = self.loss_3d(voxel_semantics, mask_camera, density_prob, semantic)
            #     losses.update(loss_occ)

            if self.nerf_head:          # 2D rendering loss
                loss_rendering = self.nerf_head(density, semantic, rays_info=rays_info, bda=lidar_aug_matrix[:,:3,:3])

                for key, val in loss_rendering.items():
                    losses['mae_part_{}_{}'.format(mae_idx, key)] = val

        if self.use_lss_depth_loss: # lss-depth loss (BEVStereo's feature)
            losses['loss_lss_depth'] = self.depth_loss
            self.depth_loss = 0


        return losses

