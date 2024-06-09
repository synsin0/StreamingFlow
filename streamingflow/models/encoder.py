import torch.nn as nn
import numpy as np
from efficientnet_pytorch import EfficientNet

from streamingflow.layers.convolutions import UpsamplingConcat, DeepLabHead

import torch.utils.checkpoint as checkpoint


class Encoder(nn.Module):
    def __init__(self, cfg, D, with_cp=False):
        super().__init__()
        self.D = D
        self.C = cfg.OUT_CHANNELS
        self.use_depth_distribution = cfg.USE_DEPTH_DISTRIBUTION
        self.downsample = cfg.DOWNSAMPLE
        self.version = cfg.NAME.split('-')[1]

        self.backbone = EfficientNet.from_pretrained(cfg.NAME)

        self.delete_unused_layers()
        if self.version == 'b4':
            self.reduction_channel = [0, 24, 32, 56, 160, 448]
        elif self.version == 'b0':
            self.reduction_channel = [0, 16, 24, 40, 112, 320]
        elif self.version == 'b7':
            self.reduction_channel = [0, 32, 48, 80, 224, 640]
        else:
            raise NotImplementedError
        self.upsampling_out_channel = [0, 48, 64, 128, 512]

        index = np.log2(self.downsample).astype(np.int)

        if self.use_depth_distribution:
            self.depth_layer_1 = DeepLabHead(self.reduction_channel[index+1], self.reduction_channel[index+1], hidden_channel=64)
            self.depth_layer_2 = UpsamplingConcat(self.reduction_channel[index+1] + self.reduction_channel[index], self.D)

        self.feature_layer_1 = DeepLabHead(self.reduction_channel[index+1], self.reduction_channel[index+1], hidden_channel=64)
        self.feature_layer_2 = UpsamplingConcat(self.reduction_channel[index+1] + self.reduction_channel[index], self.C)
        
        self.with_cp = with_cp


    def delete_unused_layers(self):
        indices_to_delete = []
        for idx in range(len(self.backbone._blocks)):
            if self.downsample == 8:
                if self.version == 'b0' and idx > 10:
                    indices_to_delete.append(idx)
                if self.version == 'b4' and idx > 21:
                    indices_to_delete.append(idx)
                if self.version == 'b7' and idx > 37:
                    indices_to_delete.append(idx)

        for idx in reversed(indices_to_delete):
            del self.backbone._blocks[idx]

        del self.backbone._conv_head
        del self.backbone._bn1
        del self.backbone._avg_pooling
        del self.backbone._dropout
        del self.backbone._fc

    def get_features_depth(self, x):
        # Adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.backbone._swish(self.backbone._bn0(self.backbone._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.backbone._blocks):
            drop_connect_rate = self.backbone._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.backbone._blocks)
            
            # x = block(x, drop_connect_rate=drop_connect_rate)
            if self.with_cp:
                x = checkpoint.checkpoint(block, x, drop_connect_rate)
            else:
                x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
                # print(len(endpoints))
                # if len(endpoints) == 3:
                #     import ipdb
                #     ipdb.set_trace()

            prev_x = x

            if self.downsample == 8:
                if self.version == 'b0' and idx == 10:
                    break
                if self.version == 'b4' and idx == 21:
                    break
                if self.version == 'b7' and idx == 37:
                    break

        # Head
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x

        index = np.log2(self.downsample).astype(np.int)
        input_1 = endpoints['reduction_{}'.format(index + 1)]
        input_2 = endpoints['reduction_{}'.format(index)]
 
        feature = self.feature_layer_1(input_1)  # torch.Size([18, 160, 20, 50])
        feature = self.feature_layer_2(feature, input_2)
        
        if self.use_depth_distribution:
            depth = self.depth_layer_1(input_1)
            depth = self.depth_layer_2(depth, input_2)
        else:
            depth = None

        return feature, depth

    def forward(self, x):
        feature, depth = self.get_features_depth(x)  # get feature vector

        # if self.use_depth_distribution:
        #     depth_prob = depth.softmax(dim=1)
        #     feature = depth_prob.unsqueeze(1) * feature.unsqueeze(2)  # outer product depth and features
        # else:
        #     feature = feature.unsqueeze(2).repeat(1, 1, self.D, 1, 1)

        return feature, depth
