import torch
import torch.nn as nn

from streamingflow.layers.temporal import Bottleneck3D, TemporalBlock
from streamingflow.layers.convolutions import ConvBlock, Bottleneck, DeepLabHead
import torch.utils.checkpoint as checkpoint

class TemporalModel(nn.Module):
    def __init__(
            self, in_channels, receptive_field, input_shape, start_out_channels=64, extra_in_channels=0,
            n_spatial_layers_between_temporal_layers=0, use_pyramid_pooling=True, with_cp = False):
        super().__init__()
        self.receptive_field = receptive_field
        n_temporal_layers = receptive_field - 1

        h, w = input_shape
        modules = []

        block_in_channels = in_channels
        block_out_channels = start_out_channels

        for _ in range(n_temporal_layers):
            if use_pyramid_pooling:
                use_pyramid_pooling = True
                pool_sizes = [(2, h, w)]
            else:
                use_pyramid_pooling = False
                pool_sizes = None
            temporal = TemporalBlock(
                block_in_channels,
                block_out_channels,
                use_pyramid_pooling=use_pyramid_pooling,
                pool_sizes=pool_sizes,
            )
            spatial = [
                Bottleneck3D(block_out_channels, block_out_channels, kernel_size=(1, 3, 3))
                for _ in range(n_spatial_layers_between_temporal_layers)
            ]
            temporal_spatial_layers = nn.Sequential(temporal, *spatial)
            modules.extend(temporal_spatial_layers)

            block_in_channels = block_out_channels
            block_out_channels += extra_in_channels

        self.out_channels = block_in_channels

        self.final_conv = DeepLabHead(block_out_channels, block_out_channels, hidden_channel=128)

        self.model = nn.Sequential(*modules)

        self.with_cp = with_cp

    def forward(self, x):
        # Reshape input tensor to (batch, C, time, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        # x = torch.cat([x,x], dim=2)

        if self.with_cp:
            x = checkpoint.checkpoint(self.model,x)
        else:
            x = self.model(x)
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        b, s, c, h, w = x.shape
        x = x.view(b * s, c, h, w)
        x = self.final_conv(x)
        x = x.view(b, s, c, h, w)

        return x


class TemporalModelIdentity(nn.Module):
    def __init__(self, in_channels, receptive_field):
        super().__init__()
        self.receptive_field = receptive_field
        self.out_channels = in_channels

    def forward(self, x):
        return x