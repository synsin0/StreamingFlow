import torch
import torch.nn as nn
import numpy as np
from streamingflow.layers.convolutions import Bottleneck, Block, DeepLabHead
from streamingflow.layers.temporal import SpatialGRU, Dual_GRU, BiGRU
from streamingflow.layers.temporal_ode_bayes import NNFOwithBayesianJumps
import copy

class FuturePredictionODE(nn.Module):
    def __init__(self, in_channels, latent_dim, n_future, cfg, mixture=True, n_gru_blocks=2, n_res_layers=1, delta_t=0.05):
        super(FuturePredictionODE, self).__init__()
        self.n_spatial_gru = n_gru_blocks
        self.delta_t = delta_t
        gru_in_channels = latent_dim
        self.gru_ode = NNFOwithBayesianJumps(input_size=in_channels, hidden_size=latent_dim, cfg=cfg, mixing=int(mixture))
        # self.dual_grus = Dual_GRU(gru_in_channels, in_channels, n_future=n_future, mixture=mixture)
        # self.res_blocks1 = nn.Sequential(*[Block(in_channels) for _ in range(n_res_layers)])
        
        self.spatial_grus = []
        self.res_blocks = []
        for i in range(self.n_spatial_gru):
            self.spatial_grus.append(SpatialGRU(in_channels, in_channels))
            if i < self.n_spatial_gru - 1:
                self.res_blocks.append(nn.Sequential(*[Block(in_channels) for _ in range(n_res_layers)]))
            else:
                self.res_blocks.append(DeepLabHead(in_channels, in_channels, 128))

        self.spatial_grus = torch.nn.ModuleList(self.spatial_grus)
        self.res_blocks = torch.nn.ModuleList(self.res_blocks)


    def forward(self, future_prediction_input, camera_states, lidar_states, camera_timestamp, lidar_timestamp,target_timestamp):
        # x has shape (b, 1, c, h, w), state: torch.Tensor [b, n_present, hidden_size, h, w]

        x_bs = []
        for bs in range(camera_states.shape[0]):
            obs_feature_with_time = {}
            if camera_states is not None:
                for index in range(camera_timestamp.shape[1]):
                    obs_feature_with_time[camera_timestamp[bs,index]] = camera_states[bs,index,:,:,:].unsqueeze(0)
            if lidar_states is not None:
                for index in range(lidar_timestamp.shape[1]):
                    obs_feature_with_time[lidar_timestamp[bs,index]] = lidar_states[bs,index,:,:,:].unsqueeze(0)
            
            obs = dict(sorted(obs_feature_with_time.items(),key = lambda v:v[0]))
            
            times = torch.tensor(list(obs.keys()))  
            observations = list(obs.values())
            observations = torch.stack(observations, dim=1)
            final_state , auxilary_loss, predict_x = self.gru_ode(times=times, input = future_prediction_input, obs = observations,delta_t=self.delta_t,T=target_timestamp[bs])
            x_bs.append(predict_x)
        
        x = torch.concat(x_bs,dim=0)


        hidden_state = x[:, 0]   # torch.Size([1, 64, 200, 200])
        for i in range(self.n_spatial_gru):
            x = self.spatial_grus[i](x, hidden_state)

            b, s, c, h, w = x.shape
            x = self.res_blocks[i](x.view(b*s, c, h, w))
            x = x.view(b, s, c, h, w)

        return x , auxilary_loss
