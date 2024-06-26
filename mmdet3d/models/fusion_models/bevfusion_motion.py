from typing import Any, Dict

import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F

from mmdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_head,
    build_neck,
    build_vtransform,
)
from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models import FUSIONMODELS


from .base import Base3DFusionModel

import matplotlib.pyplot as plt

def vis_bev_feature(bev_feat, out_path):
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)
    bev_feat = bev_feat - bev_feat.min()
    feat_map = bev_feat[-1].max(0)[0].cpu().numpy()
    im = ax.imshow(feat_map, cmap=plt.cm.jet)
    plt.axis('off')
    plt.savefig(out_path+'.jpg')


__all__ = ["StreamingFlow"]


@FUSIONMODELS.register_module()
class StreamingFlow(Base3DFusionModel):
    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__()

        self.encoders = nn.ModuleDict()
        if encoders.get("camera") is not None:
            self.use_neck = 'neck' in encoders["camera"]
            if self.use_neck:
                self.encoders["camera"] = nn.ModuleDict(
                    {
                        "backbone": build_backbone(encoders["camera"]["backbone"]),
                        "neck": build_neck(encoders["camera"]["neck"]),
                        "vtransform": build_vtransform(encoders["camera"]["vtransform"]),
                    }
                )
            else:
                self.encoders["camera"] = nn.ModuleDict(
                    {
                        "backbone": build_backbone(encoders["camera"]["backbone"]),
                        "vtransform": build_vtransform(encoders["camera"]["vtransform"]),
                    }
                )
        

        if encoders.get("lidar") is not None:
            if encoders["lidar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["lidar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders["lidar"]["voxelize"])
            self.encoders["lidar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["lidar"]["backbone"]),
                }
            )
            self.voxelize_reduce = encoders["lidar"].get("voxelize_reduce", True)

        if fuser is not None:
            self.fuser = build_fuser(fuser)
        else:
            self.fuser = None

        self.temporal_model = build_neck(encoders['temporal_model'])
        # self.decoder = nn.ModuleDict(
        #     {
        #         "backbone": build_backbone(decoder["backbone"]),
        #         "neck": build_neck(decoder["neck"]),
        #     }
        # )

        self.heads = nn.ModuleDict()
        for name in heads:
            if heads[name] is not None:
                self.heads[name] = build_head(heads[name])

        if "loss_scale" in kwargs:
            self.loss_scale = kwargs["loss_scale"]
        else:
            self.loss_scale = dict()
            for name in heads:
                if heads[name] is not None:
                    self.loss_scale[name] = 1.0

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

        x = self.encoders["camera"]["backbone"](x)
        if self.use_neck:
            x = self.encoders["camera"]["neck"](x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        x = self.encoders["camera"]["vtransform"](
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
        )
        return x

    def extract_lidar_features(self, x) -> torch.Tensor:
        feats, coords, sizes = self.voxelize(x)
        batch_size = coords[-1, 0] + 1

        import ipdb
        ipdb.set_trace()
        x = self.encoders["lidar"]["backbone"](feats, coords, batch_size, sizes=sizes)
        return x

    @torch.no_grad()
    @force_fp32()
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

    @auto_fp16(apply_to=("img", "points"))
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

    @auto_fp16(apply_to=("img", "points"))
    def forward_single(
        self,
        img, # torch.Size([1, 3, 6, 3, 256, 704])
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


        B, T, N, C ,H, W = img.shape
        img = img.view(B*T, N, C, H, W)
        batch_size = len(points)
        receptive_field = len(points[0])
        points = [tensor for sublist in points for tensor in sublist]

        camera2ego = camera2ego.flatten(0,1)
        lidar2ego = lidar2ego.flatten(0,1)
        lidar2camera = lidar2camera.flatten(0,1)
        lidar2image = lidar2image.flatten(0,1)
        camera_intrinsics = camera_intrinsics.flatten(0,1)
        camera2lidar = camera2lidar.flatten(0,1)
        img_aug_matrix = img_aug_matrix.flatten(0,1)
        lidar_aug_matrix = lidar_aug_matrix.flatten(0,1)

        features = []
        for sensor in (
            self.encoders if self.training else list(self.encoders.keys())[::-1]
        ):
            if sensor == "camera":
                feature = self.extract_camera_features(
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
            elif sensor == "lidar":
                feature = self.extract_lidar_features(points)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")
            features.append(feature)

        if not self.training:
            # avoid OOM
            features = features[::-1]
        

        # vis_bev_feature(features[1].detach(), 'before_fusion')

        if self.fuser is not None:
            x = self.fuser(features)
        else:
            assert len(features) == 1, features
            x = features[0]
        
        # vis_bev_feature(x.detach(), 'after_fusion')

        _, C, H, W = x.shape
        x = x.view(B, T, C, H, W)

        lidar_aug_matrix = lidar_aug_matrix.view(B, T, 4,4)[:,-1]
        # temporal processing
        x = self.temporal_model(x, future_egomotion=kwargs['future_egomotions'],
                                aug_transform=lidar_aug_matrix, img_is_valid=kwargs['img_is_valid'])

        batch_size = x.shape[0]
        # x = self.decoder["backbone"](x)
        # x = self.decoder["neck"](x)

        if self.training:
            outputs = {}
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                elif type == "map":
                    losses = head(x, gt_masks_bev)
                elif type == 'motion':
                    targets = kwargs
                    targets['gt_bboxes_3d'] = gt_bboxes_3d
                    targets['gt_labels_3d'] = gt_labels_3d
                    targets['gt_masks_bev'] = gt_masks_bev
                    targets['aug_transform'] = lidar_aug_matrix
                    targets['future_egomotion'] = kwargs['future_egomotions']
                    targets['img_is_valid'] = kwargs['img_is_valid']

                    predictions = head(x, targets, **kwargs)
                    losses = head.loss(predictions)
                else:
                    raise ValueError(f"unsupported head: {type}")
                for name, val in losses.items():
                    if val.requires_grad:
                        outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]
                    else:
                        outputs[f"stats/{type}/{name}"] = val
            return outputs
        else:
            outputs = [{} for _ in range(batch_size)]
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    bboxes = head.get_bboxes(pred_dict, metas)
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
                elif type == "map":
                    logits = head(x)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
                elif type == "motion":
                    targets = kwargs
                    targets['gt_bboxes_3d'] = gt_bboxes_3d
                    targets['gt_labels_3d'] = gt_labels_3d
                    targets['gt_masks_bev'] = gt_masks_bev
                    targets['aug_transform'] = lidar_aug_matrix
                    targets['future_egomotion'] = kwargs['future_egomotions']
                    targets['img_is_valid'] = kwargs['img_is_valid']

                    predictions = head(x, targets, **kwargs)             
                    seg_prediction, pred_consistent_instance_seg = head.inference(predictions)
                    outputs[0]['motion_predictions'] = predictions
                    outputs[0]['motion_segmentation'] = seg_prediction
                    outputs[0]['motion_instance'] = pred_consistent_instance_seg
                else:
                    raise ValueError(f"unsupported head: {type}")
            return outputs
