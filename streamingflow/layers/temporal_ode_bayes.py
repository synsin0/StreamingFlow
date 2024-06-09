import torch
import math
import numpy as np
# from torchdiffeq import odeint
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from streamingflow.layers.convolutions import ConvBlock
from streamingflow.utils.geometry import warp_features
from streamingflow.layers.res_models import SmallEncoder, SmallDecoder, ConvNet
from streamingflow.models.distributions import DistributionModule
from streamingflow.models import model_utils
from streamingflow.layers.convolutions import Bottleblock
# GRU-ODE: Neural Negative Feedback ODE with Bayesian jumps

class SpatialGRUODECell(torch.nn.Module):
    def __init__(self, input_size, hidden_size,gru_bias_init=0.0, norm='bn', activation='relu', bias=True):
        """
        For p(t) modelling input_size should be 2x the x size.
        """
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.bias        = bias

        self.gru_bias_init = gru_bias_init

        self.conv_update = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_reset = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)

        self.conv_state_tilde = ConvBlock(
            input_size + hidden_size, hidden_size, kernel_size=3, bias=False, norm=norm, activation=activation
        )
        


    def forward(self, x, state):
        """
        Returns a change due to one step of using GRU-ODE for all state.
        The step size is given by delta_t.

        Args:
            x        input values
            state        hidden state (current)
            delta_t  time step

        Returns:
            Updated state
        """

      
        x_and_state = torch.cat([x, state], dim=1)
        update_gate = self.conv_update(x_and_state)
        reset_gate = self.conv_reset(x_and_state)
        # Add bias to initialise gate as close to identity function
        update_gate = torch.sigmoid(update_gate + self.gru_bias_init)
        reset_gate = torch.sigmoid(reset_gate + self.gru_bias_init)

        # Compute proposal state, activation is defined in norm_act_config (can be tanh, ReLU etc)
        state_tilde = self.conv_state_tilde(torch.cat([x, (1.0 - reset_gate) * state], dim=1))

        dh = update_gate * (state_tilde - state)
        return dh


class DualGRUODECell(torch.nn.Module):
    def __init__(self, input_size, hidden_size,gru_bias_init=0.0, norm='bn', activation='relu', bias=True):
        """
        For p(t) modelling input_size should be 2x the x size.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size


        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru_bias_init = gru_bias_init

        self.conv_update_1 = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_reset_1 = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_state_tilde_1 = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)

        self.conv_update_2 = nn.Conv2d(hidden_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_reset_2 = nn.Conv2d(hidden_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_state_tilde_2 = nn.Conv2d(hidden_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_decoder_2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)

        self.trusting_gate = nn.Sequential(
            Bottleblock(hidden_size+hidden_size, hidden_size),
            nn.Conv2d(hidden_size, 2, kernel_size=1, bias=False)
        )

    def forward(self, x, state):
        '''
        x: torch.Tensor [b, 1, input_size, h, w]
        state: torch.Tensor [b, n_present, hidden_size, h, w]
        '''
        if len(x.shape)==4:
            x = x.unsqueeze(0)
            state = state.unsqueeze(0)
        b, s, c, h, w = x.shape
        assert c == self.input_size, f'feature sizes must match, got input {c} for layer with size {self.input_size}'
        n_present = state.shape[1]

        h = state[:, 0]

        # warmup
        for t in range(n_present - 1):
            cur_state = state[:, t]
            h = self.gru_cell_2(cur_state, h)

        # recurrent layers
        rnn_state1 = state[:, -1]
        rnn_state2 = state[:, -1]
        x = x[:, 0]

        
        # propagate gru v1
        rnn_state1 = self.gru_cell_1(x, rnn_state1)
        # propagate gru v2
        h = self.gru_cell_2(rnn_state2, h)
        rnn_state2 = self.conv_decoder_2(h)

        # mix the two distribution
        mix_state = torch.cat([rnn_state1, rnn_state2], dim=1)
        trust_gate = self.trusting_gate(mix_state)
        trust_gate = torch.softmax(trust_gate, dim=1)
        cur_state = rnn_state2 * trust_gate[:,0:1] + rnn_state1 * trust_gate[:,1:]
        


        return cur_state - state.squeeze(1)

    def gru_cell_1(self, x, state):
        # Compute gates
        x_and_state = torch.cat([x, state], dim=1)
        update_gate = self.conv_update_1(x_and_state)
        reset_gate = self.conv_reset_1(x_and_state)
        # Add bias to initialise gate as close to identity function
        update_gate = torch.sigmoid(update_gate + self.gru_bias_init)
        reset_gate = torch.sigmoid(reset_gate + self.gru_bias_init)

        # Compute proposal state, activation is defined in norm_act_config (can be tanh, ReLU etc)
        state_tilde = self.conv_state_tilde_1(torch.cat([x, (1.0 - reset_gate) * state], dim=1))

        output = (1.0 - update_gate) * state + update_gate * state_tilde
        return output

    def gru_cell_2(self, x, state):
        # Compute gates
        x_and_state = torch.cat([x, state], dim=1)
        update_gate = self.conv_update_2(x_and_state)
        reset_gate = self.conv_reset_2(x_and_state)
        # Add bias to initialise gate as close to identity function
        update_gate = torch.sigmoid(update_gate + self.gru_bias_init)
        reset_gate = torch.sigmoid(reset_gate + self.gru_bias_init)

        # Compute proposal state, activation is defined in norm_act_config (can be tanh, ReLU etc)
        state_tilde = self.conv_state_tilde_2(torch.cat([x, (1.0 - reset_gate) * state], dim=1))

        output = (1.0 - update_gate) * state + update_gate * state_tilde
        return output



class SpatialGRUCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size,gru_bias_init=0.0, norm='bn', activation='relu', bias=True):
        """
        For p(t) modelling input_size should be 2x the x size.
        """
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.bias        = bias

        self.gru_bias_init = gru_bias_init

        self.conv_update = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_reset = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)

        self.conv_state_tilde = ConvBlock(
            input_size + hidden_size, hidden_size, kernel_size=3, bias=False, norm=norm, activation=activation
        )

    def forward(self, x, state):
        """
        Returns a change due to one step of using GRU-ODE for all state.
        The step size is given by delta_t.

        Args:
            x        input values
            state        hidden state (current)
            delta_t  time step

        Returns:
            Updated state
        """
        x_and_state = torch.cat([x, state], dim=1)
        update_gate = self.conv_update(x_and_state)
        reset_gate = self.conv_reset(x_and_state)
        # Add bias to initialise gate as close to identity function
        update_gate = torch.sigmoid(update_gate + self.gru_bias_init)
        reset_gate = torch.sigmoid(reset_gate + self.gru_bias_init)

        # Compute proposal state, activation is defined in norm_act_config (can be tanh, ReLU etc)
        state_tilde = self.conv_state_tilde(torch.cat([x, (1.0 - reset_gate) * state], dim=1))

        output = (1.0 - update_gate) * state + update_gate * state_tilde
        return output


class DualGRUCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size,gru_bias_init=0.0, norm='bn', activation='relu', bias=True):
        """
        For p(t) modelling input_size should be 2x the x size.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size


        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru_bias_init = gru_bias_init

        self.conv_update_1 = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_reset_1 = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_state_tilde_1 = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)

        self.conv_update_2 = nn.Conv2d(hidden_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_reset_2 = nn.Conv2d(hidden_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_state_tilde_2 = nn.Conv2d(hidden_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_decoder_2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)

        self.trusting_gate = nn.Sequential(
            Bottleblock(hidden_size+hidden_size, hidden_size),
            nn.Conv2d(hidden_size, 2, kernel_size=1, bias=False)
        )

    def forward(self, x, state):
        '''
        x: torch.Tensor [b, 1, input_size, h, w]
        state: torch.Tensor [b, n_present, hidden_size, h, w]
        '''
 
        if len(x.shape)==4:
            x = x.unsqueeze(0)
            state = state.unsqueeze(0)
        b, s, c, h, w = x.shape
        assert c == self.input_size, f'feature sizes must match, got input {c} for layer with size {self.input_size}'
        n_present = state.shape[1]

        h = state[:, 0]


        # recurrent layers
        rnn_state1 = state[:, -1]
        rnn_state2 = state[:, -1]
        x = x[:, 0]

        
        # propagate gru v1
        rnn_state1 = self.gru_cell_1(x, rnn_state1)
        # propagate gru v2
        h = self.gru_cell_2(rnn_state2, h)
        rnn_state2 = self.conv_decoder_2(h)

        # mix the two distribution
        mix_state = torch.cat([rnn_state1, rnn_state2], dim=1)
        trust_gate = self.trusting_gate(mix_state)
        trust_gate = torch.softmax(trust_gate, dim=1)
        cur_state = rnn_state2 * trust_gate[:,0:1] + rnn_state1 * trust_gate[:,1:]



        return cur_state

    def gru_cell_1(self, x, state):
        # Compute gates
        x_and_state = torch.cat([x, state], dim=1)
        update_gate = self.conv_update_1(x_and_state)
        reset_gate = self.conv_reset_1(x_and_state)
        # Add bias to initialise gate as close to identity function
        update_gate = torch.sigmoid(update_gate + self.gru_bias_init)
        reset_gate = torch.sigmoid(reset_gate + self.gru_bias_init)

        # Compute proposal state, activation is defined in norm_act_config (can be tanh, ReLU etc)
        state_tilde = self.conv_state_tilde_1(torch.cat([x, (1.0 - reset_gate) * state], dim=1))

        output = (1.0 - update_gate) * state + update_gate * state_tilde
        return output

    def gru_cell_2(self, x, state):
        # Compute gates
        x_and_state = torch.cat([x, state], dim=1)
        update_gate = self.conv_update_2(x_and_state)
        reset_gate = self.conv_reset_2(x_and_state)
        # Add bias to initialise gate as close to identity function
        update_gate = torch.sigmoid(update_gate + self.gru_bias_init)
        reset_gate = torch.sigmoid(reset_gate + self.gru_bias_init)

        # Compute proposal state, activation is defined in norm_act_config (can be tanh, ReLU etc)
        state_tilde = self.conv_state_tilde_2(torch.cat([x, (1.0 - reset_gate) * state], dim=1))

        output = (1.0 - update_gate) * state + update_gate * state_tilde
        return output


class GRUObservationCell(torch.nn.Module):
    """Implements discrete update based on the received observations."""

    def __init__(self, input_size, hidden_size,min_log_sigma=-5.0,max_log_sigma=5.0, bias=True):
        super().__init__()
        self.gru_d     = DualGRUCell(input_size, hidden_size, bias=bias)
        prep_hidden = hidden_size
        ## prep layer and its initialization
        std            = math.sqrt(2.0 / (4 + prep_hidden))


        self.input_size  = input_size
        self.prep_hidden = prep_hidden
        self.var_eps     = 1e-6
        self.min_log_sigma = min_log_sigma
        self.max_log_sigma = max_log_sigma
        # self.present_distribution = DistributionModule(
        #             self.input_size,
        #             self.prep_hidden,
        #             method='GAUSSIAN',
        #         )


    def get_mu_sigma(self,mu_log_sigma):
        mu = mu_log_sigma[:, :, :self.prep_hidden]
        log_sigma = mu_log_sigma[:, :, self.prep_hidden:2*self.prep_hidden]
        log_sigma = torch.clamp(log_sigma, self.min_log_sigma, self.max_log_sigma)
        sigma = torch.exp(log_sigma)
        # if self.training:
        #     gaussian_noise = torch.randn((b, s, self.prep_hidden), device=present_features.device)
        # else:
        #     gaussian_noise = torch.zeros((b, s, self.prep_hidden), device=present_features.device)
        # sample = mu + sigma * gaussian_noise
        return mu, log_sigma, sigma

    def forward(self, state, p, X_obs):


        # mu_log_sigma_p = self.present_distribution(p_obs.unsqueeze(1))
        # present_mu_p, present_log_sigma_p, sigma_p = self.get_mu_sigma(mu_log_sigma_p)
        
        # mu_log_sigma_obs = self.present_distribution(X_obs.unsqueeze(1))
        # present_mu_obs, present_log_sigma_obs, sigma_obs = self.get_mu_sigma(mu_log_sigma_obs)

        # # mean, var = torch.chunk(p_obs, 2, dim=1)
        # ## making var non-negative and also non-zero (by adding a small value)
        # mean      = present_mu_p
        # var       = torch.abs(sigma_p) + self.var_eps
        # error     = (present_mu_obs - present_mu_p) / torch.sqrt(var)
    
        # ## log normal loss, over all observations
        # loss         = 0.5 * ((torch.pow(error, 2) + torch.log(var))).sum()

        
        bs, C, h, w = X_obs.shape


        # mean = mean.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,h,w)
        # var = var.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,h,w)
        # error = error.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,h,w)


        ## TODO: try removing X_obs (they are included in error)
        gru_input = X_obs
        # gru_input = self.conv_fuser([X_obs,p_obs])
        # gru_input    = torch.stack([X_obs.unsqueeze(1), mean, var, error], dim=2).squeeze(1).view(bs,-1,h,w)
        # gru_input = self.obs_model(gru_input)

        # gru_input    = torch.matmul(gru_input, self.w_prep).squeeze(2) + self.bias_prep
        # gru_input.relu_()
        # ## gru_input is (sample x feature x prep_hidden)
        # gru_input    = gru_input.permute(2, 0, 1)
        # gru_input    = gru_input.permute(1, 2, 0).contiguous().view(-1, self.prep_hidden * self.input_size)

      
        state = self.gru_d(gru_input, state)
      
        loss= None
        return state, loss




def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.05)

class NNFOwithBayesianJumps(torch.nn.Module):
    ## Neural Negative Feedback ODE with Bayesian jumps
    def __init__(self, input_size, hidden_size, cfg, bias=True, logvar=True, mixing=1,solver="euler",min_log_sigma=-5.0,max_log_sigma=5.0, impute = False):
        """
        The smoother variable computes the classification loss as a weighted average of the projection of the latents at each observation.
        impute feeds the parameters of the distribution to GRU-ODE at each step.
        """

        super().__init__()

        self.impute = cfg.MODEL.IMPUTE
        self.cfg = cfg

        self.min_log_sigma = min_log_sigma
        self.max_log_sigma = max_log_sigma
        self.p_model = ConvNet(hidden_size , hidden_size * 2,)

        # self.srvp_encoder = SmallEncoder(hidden_size, hidden_size,
        #                                  self.cfg.MODEL.SMALL_ENCODER.FILTER_SIZE)
        # self.srvp_decoder = SmallDecoder(hidden_size, hidden_size,
        #                                  self.cfg.MODEL.SMALL_ENCODER.FILTER_SIZE, self.skipco)

        # self.classification_model = torch.nn.Sequential(
        #     torch.nn.Linear(hidden_size,classification_hidden,bias=bias),
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout(p=dropout_rate),
        #     torch.nn.Linear(classification_hidden,1,bias=bias)
        # )
        
        # if full_gru_ode:
        #     if impute is False:
        #         self.gru_c   = FullGRUODECell_Autonomous(hidden_size, bias = bias)
        #     else:
        #         self.gru_c   = FullGRUODECell(2 * input_size, hidden_size, bias=bias)

        # else:
        #     if impute is False:
        #         self.gru_c = GRUODECell_Autonomous(hidden_size, bias= bias)
        #     else:
        #         self.gru_c   = GRUODECell(2 * input_size, hidden_size, bias=bias)
        self.gru_c   = DualGRUODECell(input_size, hidden_size, bias=bias)

        # if logvar:
        #     self.gru_obs = GRUObservationCellLogvar(input_size, hidden_size, prep_hidden, bias=bias)
        # else:

        self.gru_obs = GRUObservationCell(input_size, hidden_size,min_log_sigma = self.min_log_sigma,max_log_sigma = self.max_log_sigma, bias=bias)
        self.skipco = self.cfg.MODEL.SMALL_ENCODER.SKIPCO
        self.srvp_encoder = SmallEncoder(self.cfg.MODEL.ENCODER.OUT_CHANNELS, self.cfg.MODEL.ENCODER.OUT_CHANNELS,
                                         self.cfg.MODEL.SMALL_ENCODER.FILTER_SIZE)
        self.srvp_decoder = SmallDecoder(self.cfg.MODEL.ENCODER.OUT_CHANNELS, self.cfg.MODEL.ENCODER.OUT_CHANNELS,
                                         self.cfg.MODEL.SMALL_ENCODER.FILTER_SIZE, self.cfg.MODEL.SMALL_ENCODER.SKIPCO)

        # self.covariates_map = torch.nn.Sequential(
        #     torch.nn.Linear(cov_size, cov_hidden, bias=bias),
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout(p=dropout_rate),
        #     torch.nn.Linear(cov_hidden, hidden_size, bias=bias),
        #     torch.nn.Tanh()
        # )


        # self.solver     = solver
        self.solver = cfg.MODEL.SOLVER
        self.use_variable_ode_step = cfg.MODEL.FUTURE_PRED.USE_VARIABLE_ODE_STEP
        assert self.solver in ["euler", "midpoint", "dopri5"], "Solver must be either 'euler' or 'midpoint' or 'dopri5'."

        # self.store_hist = options.pop("store_hist",False)
        self.input_size = input_size
        self.logvar     = logvar
        self.mixing     = mixing # mixing hyperparameter for loss_1 and loss_2 aggregation.

        self.apply(init_weights)


    def srvp_decode(self, x, skip=None):
        # content vector is missing. 
        """
        Decodes SRVP intermediate states into LSS output space
        x: SRVP intermediate states, torch.Tensor, [batch, seq_len, channels, height, width]
        Returns:
        torch.Tensor: [batch, seq_len, channels, height, width]
        """
        b, t, c, h, w = x.shape
        _x = x.reshape(b * t, c, h, w)
        if skip:
            skip = [s.unsqueeze(1).expand(b, t, *s.shape[1:]) for s in skip]
            skip = [s.reshape(t * b, *s.shape[2:]) for s in skip]
        x_out = self.srvp_decoder(_x, skip=skip)
        return x_out.view(b, t, *x_out.shape[1:])

    def srvp_encode(self, x):
        """
        Encodes LSS's outputs
            x: LSS encoded features, torch.Tensor, [batch, seq_len, channels, height, width]
        Returns:
            torch.Tensor: [batch, seq_len, channels, height, width]
        """
        b, t, c, h, w = x.shape
        _x = x.view(b * t, c, h, w)
        hx, skips = self.srvp_encoder(_x, return_skip=True)
        hx = hx.view(b, t, *hx.shape[1:])
        if self.skipco:
            if self.training:
                # When training, take a random frame to compute the skip connections
                tx = torch.randint(t, size=(b,)).to(hx.device)
                index = torch.arange(b).to(hx.device)
                skips = [s.view(b, t, *s.shape[1:])[index, tx] for s in skips]
            else:
                # When testing, choose the last frame
                skips = [s.view(b, t, *s.shape[1:])[:, -1] for s in skips]
        else:
            skips = None
        return hx, skips

    def ode_step(self, state, input, delta_t, current_time):
      
        """Executes a single ODE step."""
        eval_times = torch.tensor([0],device = state.device, dtype = torch.float64)
        eval_ps = torch.tensor([0],device = state.device, dtype = torch.float32)

        if self.impute is False:
            input = torch.zeros_like(input)
          
        if self.solver == "euler":
            state = state + delta_t * self.gru_c(input, state)
            input = self.infer_state(state)[0]


        elif self.solver == "midpoint":
            k  = state + delta_t / 2 * self.gru_c(input, state)
            pk = self.infer_state(k)[0]

            state = state + delta_t * self.gru_c(pk, k)
            input = self.infer_state(state)[0]

        # elif self.solver == "dopri5":
        #     assert self.impute==False #Dopri5 solver is only compatible with autonomous ODE.
        #     solution, eval_times, eval_vals = odeint(self.gru_c,state,torch.tensor([0,delta_t]),method=self.solver,options={"store_hist":self.store_hist})
        #     if self.store_hist:
        #         eval_ps = self.p_model(torch.stack([ev[0] for ev in eval_vals]))
        #     eval_times = torch.stack(eval_times) + current_time
        #     state = solution[1,:,:]
        #     input = self.p_model(state)
        
        current_time += delta_t
        return state, input, current_time, eval_times, eval_ps

        raise ValueError(f"Unknown solver '{self.solver}'.")

    def infer_state(self, x, deterministic=False):
        """
        Creates the first state from the conditioning observation
            x: encoded features, torch.Tensor, [batch, seq_len, channels, height, width]
            deterministic: If true, it will return only the means otherwise a sample from Normal distribution
        Returns:
            First state of the model -> y, torch.Tensor, [batch, channels, height, width]
        """
        # Q1: will the first state be stochastic?
        # Q2: are we going to sample a different noise for each position

        q_y0_params = self.p_model(x)
        y_0 = model_utils.rsample_normal(q_y0_params, max_log_sigma=self.max_log_sigma,
                                         min_log_sigma=self.min_log_sigma)
        return y_0, q_y0_params

    def forward(self, times, input, obs, delta_t, T, return_path=True,):
        """
        Args:
            times      np vector of observation times
            time_ptr   start indices of data for a given time
            X          data tensor
            M          mask tensor (1.0 if observed, 0.0 if unobserved)
            obs_idx    observed patients of each datapoint (indexed within the current minibatch)
            delta_t    time step for Euler
            T          total time
            cov        static covariates for learning the first h0
            return_path   whether to return the path of state

        Returns:
            state          hidden state at final time (T)
            loss       loss of the Gaussian observations
        """

        # state = self.covariates_map(cov)

        # p            = self.p_model(state)
        # vis_bev_feature(obs,'obs')

        hx_obs, skips_obs = self.srvp_encode(obs)
        input, input_obs = self.srvp_encode(input)
        bs, seq, c, h, w = input.shape  # [1, 1, 64, 200, 200]
        input = input.view(bs * seq, c, h, w)

        state = torch.zeros_like(input)  # constant init of temporal state
        current_time = times.min().item()

        counter      = 0

        loss_pre_jump = 0 #Pre-jump loss
        loss_post_jump = 0 #Post-jump loss (KL between p_updated and the actual sample)

        path_t = []
        path_h = []
        path_p = []
        # if return_path:
        #     path_t = [0]
        #     path_p = [input]
        #     path_h = [state]

        # if smoother:
        #     class_loss_vec = torch.zeros(cov.shape[0],device = state.device)
        #     num_evals_vec  = torch.zeros(cov.shape[0],device = state.device)
        #     class_criterion = class_criterion
        #     assert class_criterion is not None

        # assert len(times) + 1 == len(time_ptr)

        
        # assert (len(times) == 0) or (times[-1] <= max(T))

        eval_times_total = torch.tensor([],dtype = torch.float64, device = state.device)
        eval_vals_total  = torch.tensor([],dtype = torch.float32, device = state.device)
        
        

        for i, obs_time in enumerate(times):
            ## Propagation of the ODE until next observation
            while current_time <= (obs_time-delta_t): #0.0001 delta_t used for numerical consistency.
                 
                if self.solver == "dopri5":
                    state, input, current_time, eval_times, eval_ps = self.ode_step(state, input, obs_time-current_time, current_time)
                else:
                    if self.use_variable_ode_step:
                        state, input, current_time, eval_times, eval_ps = self.ode_step(state, input, obs_time-current_time, current_time)
                    else:
                        state, input, current_time, eval_times, eval_ps = self.ode_step(state, input, delta_t, current_time)
                eval_times_total = torch.cat((eval_times_total, eval_times))
                eval_vals_total  = torch.cat((eval_vals_total, eval_ps))
                if isinstance(current_time,torch.Tensor):
                    current_time = current_time.item()
                # if current_time < obs_time:
                    #Storing the predictions.
                    # if return_path:
                    #     path_t.append(current_time)
                    #     path_p.append(input)
                    #     path_h.append(state)

            ## Reached an observation
            X_obs = hx_obs[:,i,:,:,:]

            ## Using GRUObservationCell to update state. Also updating input and loss
            state, losses = self.gru_obs(state, input, X_obs)
           
            # if smoother:
            #     class_loss_vec[i_obs] += class_criterion(self.classification_model(state[i_obs]),labels[i_obs]).squeeze(1)
            #     num_evals_vec[i_obs] +=1
            # if losses.sum()!=losses.sum():

            # loss_pre_jump    = loss_pre_jump+ losses.sum()

            input         = self.infer_state(state)[0]

            # loss_post_jump = loss_post_jump + compute_KL_loss(p_obs = input, X_obs = X_obs, logvar=self.logvar)
            
            if return_path:
                path_t.append(obs_time.item())
                # path_p.append(input)
                path_h.append(state)

        # current_time = 1.0
        ## after every observation has been processed, propagating until T
        for predict_time in T:
            while current_time < predict_time:
                if self.solver == "dopri5":
                    state, input, current_time,eval_times, eval_ps = self.ode_step(state, input, predict_time-current_time, current_time)
                else:
                    if self.use_variable_ode_step:
                        state, input, current_time,eval_times, eval_ps = self.ode_step(state, input, predict_time-current_time, current_time)
                    else:
                        state, input, current_time,eval_times, eval_ps = self.ode_step(state, input, delta_t, current_time)
                eval_times_total = torch.cat((eval_times_total,eval_times))
                eval_vals_total  = torch.cat((eval_vals_total, eval_ps))
                #counter += 1
                #current_time = counter * delta_t
                if isinstance(current_time,torch.Tensor):
                    current_time = current_time.item()
                #Storing the predictions
                if current_time > predict_time - 0.5 * delta_t and current_time < predict_time + 0.5 * delta_t :
                    path_t.append(current_time)
                    # path_p.append(input)
                    path_h.append(state)

        x = []
        path_t = np.array(path_t)
        

        for time_stamp in T:
            if isinstance(time_stamp, torch.Tensor):
                time_stamp = time_stamp.item()
            A = np.where(path_t > time_stamp -  0.5 * delta_t)[0]
            B = np.where(path_t < time_stamp +  0.5 * delta_t)[0]

            if np.any(np.in1d(A, B)):
                idx = np.max(A[np.in1d(A,B)])
            else:
                idx = np.argmin(np.abs(path_t - time_stamp))
            x.append(path_h[idx])
    
        x = torch.stack(x,dim=1)
        
        x = self.srvp_decode(x)
        

       
        loss = 0
        return state, loss, x

# def compute_KL_loss(p_obs, X_obs, obs_noise_std=1e-2, logvar=True):
#     obs_noise_std = torch.tensor(obs_noise_std)
#     if logvar:
#         mean, var = torch.chunk(p_obs, 2, dim=1)
#         std = torch.exp(0.5*var)
#     else:
#         mean, var = torch.chunk(p_obs, 2, dim=1)
#         ## making var non-negative and also non-zero (by adding a small value)
#         std       = torch.pow(torch.abs(var) + 1e-5,0.5)

#     return (gaussian_KL(mu_1 = mean, mu_2 = X_obs, sigma_1 = std, sigma_2 = obs_noise_std)).sum()


# def gaussian_KL(mu_1, mu_2, sigma_1, sigma_2):
#     return(torch.log(sigma_2) - torch.log(sigma_1) + (torch.pow(sigma_1,2)+torch.pow((mu_1 - mu_2),2)) / (2*sigma_2**2) - 0.5)


# class GRUODEBayesSeq(torch.nn.Module):
#     def __init__(self, input_size, hidden_size, p_hidden, prep_hidden, bias=True, cov_size=1, cov_hidden=1, classification_hidden=1, logvar=True, mixing=1, dropout_rate=0, obs_noise_std=1e-2, full_gru_ode=False):
#         super().__init__()
#         self.obs_noise_std = obs_noise_std
#         self.classification_model = torch.nn.Sequential(
#             torch.nn.Linear(hidden_size, classification_hidden, bias=bias),
#             torch.nn.ReLU(),
#             torch.nn.Dropout(p=dropout_rate),
#             torch.nn.Linear(classification_hidden, 1, bias=bias),
#         )
#         # if full_gru_ode:
#         #     self.gru_c   = FullGRUODECell(2 * input_size, hidden_size, bias=bias)
#         # else:
#         #     self.gru_c   = GRUODECell(2 * input_size, hidden_size, bias=bias)
#         self.gru_c   = SpatialGRUODECell(2 * input_size, hidden_size, bias=bias)
#         self.gru_bayes = SeqGRUBayes(input_size=input_size, hidden_size=hidden_size, prep_hidden=prep_hidden, p_hidden=p_hidden, bias=bias)

#         self.covariates_map = torch.nn.Sequential(
#             torch.nn.Linear(cov_size, cov_hidden, bias=bias),
#             torch.nn.ReLU(),
#             torch.nn.Dropout(p=dropout_rate),
#             torch.nn.Linear(cov_hidden, hidden_size, bias=bias),
#         )
#         self.input_size = input_size
#         self.mixing     = mixing #mixing hyperparameter for loss_1 and loss_2 aggregation.
#         self.apply(init_weights)

#     def forward(self, times, time_ptr, Xpadded, Fpadded, X, M, lengths,
#                 obs_idx, delta_t, T, cov, return_path=False):
#         """
#         Args:
#             times      np vector of observation times
#             time_ptr   start indices of data for a given time
#             Xpadded    data tensor (padded)
#             Fpadded    feature id of each data point (padded)
#             X          observation tensor
#             M          mask tensor
#             obs_idx    observed patients of each datapoint (current minibatch)
#             delta_t    time step for Euler
#             T          total time
#             cov        static covariates for learning the first h0
#             return_path   whether to return the path of state

#         Returns:
#             state          hidden state at final time (T)
#             loss       loss of the Gaussian observations
#         """

#         state       = self.covariates_map(cov)
#         p       = self.gru_bayes.p_model(state)
#         time    = 0.0
#         counter = 0

#         loss_1 = 0 # Pre-jump loss
#         loss_2 = 0 # Post-jump loss

#         if return_path:
#             path_t = [0]
#             path_p = [p]

#         assert len(times) + 1 == len(time_ptr)
#         assert (len(times) == 0) or (times[-1] <= T)

#         for i, obs_time in enumerate(times):
#             ## Propagation of the ODE until obs_time
#             while time < obs_time:
#                 state = state + delta_t * self.gru_c(p, state)
#                 p = self.gru_bayes.p_model(state)

#                 ## using counter to avoid numerical errors
#                 counter += 1
#                 time = counter * delta_t
#                 ## Storing the predictions.
#                 if return_path:
#                     path_t.append(time)
#                     path_p.append(p)

#             ## Reached obs_time
#             start = time_ptr[i]
#             end   = time_ptr[i+1]

#             L_obs = lengths[start:end]
#             X_obs = pack_padded_sequence(Xpadded[start:end], L_obs, batch_first=True)
#             F_obs = pack_padded_sequence(Fpadded[start:end], L_obs, batch_first=True)
#             i_obs = obs_idx[start:end]

#             Xf_batch = X[start:end]
#             Mf_batch = M[start:end]

#             ## Using GRU-Bayes to update state. Also updating p and loss
#             state, loss_i, loss_pre = self.gru_bayes(state, X_obs, F_obs, i_obs, X=Xf_batch, M=Mf_batch)
#             loss_1    = loss_1 + loss_i + loss_pre.sum()
#             p         = self.gru_bayes.p_model(state)

#             loss_2 = loss_2 + compute_KL_loss(p_obs = p[i_obs], X_obs = Xf_batch, M_obs = Mf_batch, obs_noise_std=self.obs_noise_std)

#             if return_path:
#                 path_t.append(obs_time)
#                 path_p.append(p)

#         while time < T:
#             state = state + delta_t * self.gru_c(p, state)
#             p = self.gru_bayes.p_model(state)

#             counter += 1
#             time = counter * delta_t
#             if return_path:
#                 path_t.append(time)
#                 path_p.append(p)

#         loss = loss_1 + self.mixing * loss_2
#         class_pred = self.classification_model(state)
#         if return_path:
#             return state, loss, class_pred, np.array(path_t), torch.stack(path_p)
#         return state, loss, class_pred


# class SeqGRUBayes(torch.nn.Module):
#     """

#     Inputs to forward:
#         state      tensor of hiddens
#         X_obs  PackedSequence of observation values
#         F_obs  PackedSequence of feature ids
#         i_obs  indices of state that have been observed

#     Returns updated state.
#     """
#     def __init__(self, input_size, hidden_size, prep_hidden, p_hidden, bias=True):
#         super().__init__()
#         self.p_model = torch.nn.Sequential(
#             torch.nn.Linear(hidden_size, p_hidden, bias=bias),
#             torch.nn.ReLU(),
#             torch.nn.Linear(p_hidden, 2 * input_size, bias=bias),
#         )
#         self.gru = SpatialGRUCell(prep_hidden, hidden_size, bias=bias)

#         ## prep layer and its initialization
#         std            = math.sqrt(2.0 / (4 + prep_hidden))
#         self.w_prep    = torch.nn.Parameter(std * torch.randn(input_size, 4, prep_hidden))
#         self.bias_prep = torch.nn.Parameter(0.1 + torch.zeros(input_size, prep_hidden))

#         self.input_size  = input_size
#         self.prep_hidden = prep_hidden
#         self.var_eps     = 1e-6

#     def p_mean_logvar(self, state):
#         p      = self.p_model(state)
#         mean, logvar = torch.chunk(p, 2, dim=1)
#         return mean, logvar

#     def step_1feature(self, hidden, X_step, F_step):
#         ## 1) Compute error on the observed features
#         mean, logvar = self.p_mean_logvar(hidden)
#         ## mean, logvar both are [ Batch  x  input_size ]
#         hrange = torch.arange(hidden.shape[0])
#         mean   = mean[   hrange, F_step ]
#         logvar = logvar[ hrange, F_step ]

#         sigma  = torch.exp(0.5 * logvar)
#         error  = (X_step - mean) / sigma

#         ## log normal loss, over all observations
#         loss   = 0.5 * (torch.pow(error, 2) + logvar).sum()

#         ## TODO: try removing X_obs (they are included in error)
#         gru_input = torch.stack([X_step, mean, logvar, error], dim=1).unsqueeze(1)
#         ## 2) Select the matrices from w_prep and bias_prep; multiply
#         W         = self.w_prep[F_step, :, :]
#         bias      = self.bias_prep[F_step]
#         gru_input = torch.matmul(gru_input, W).squeeze(1) + bias
#         gru_input.relu_()

#         return self.gru(gru_input, hidden), loss

#     def ode_step(self, state, p, delta_t):
#         if self.solver == "euler":
#             state = state + delta_t * self.gru_c(p, state)
#             p = self.p_model(state)
#             return state, p

#         if self.solver == "midpoint":
#             k  = state + delta_t / 2 * self.gru_c(p, state)
#             pk = self.p_model(k)

#             h2 = state + delta_t * self.gru_c(pk, k)
#             p2 = self.p_model(h2)
#             return h2, p2

#         raise ValueError(f"Unknown solver '{self.solver}'.")

#     def forward(self, state, X_obs, F_obs, i_obs, X, M):
#         """
#         See https://github.com/pytorch/pytorch/blob/a462edd0f6696a4cac4dd04c60d1ad3c9bc0b99c/torch/nn/_functions/rnn.py#L118-L154
#         """
#         ## selecting state to be updated
#         hidden = state[i_obs]

#         output          = []
#         input_offset    = 0
#         last_batch_size = X_obs.batch_sizes[0]
#         hiddens         = []
#         #flat_hidden     = not isinstance(hidden, tuple)


#         ## computing loss before any updates
#         mean, logvar = self.p_mean_logvar(hidden)
#         sigma        = torch.exp(0.5 * logvar)
#         error        = (X - mean) / sigma
#         losses_pre   = 0.5 * ((torch.pow(error, 2) + logvar) * M)

#         ## updating hidden
#         loss = 0
#         input_offset = 0
#         for batch_size in X_obs.batch_sizes:
#             X_step = X_obs.data[input_offset:input_offset + batch_size]
#             F_step = F_obs.data[input_offset:input_offset + batch_size]
#             input_offset += batch_size

#             dec = last_batch_size - batch_size
#             if dec > 0:
#                 hiddens.append(hidden[-dec:])
#                 hidden = hidden[:-dec]
#             last_batch_size = batch_size

#             hidden, loss_b = self.step_1feature(hidden, X_step, F_step)
#             loss = loss + loss_b

#         hiddens.append(hidden)
#         hiddens.reverse()

#         hidden = torch.cat(hiddens, dim=0)

#         ## updating observed trajectories
#         h2        = state.clone()
#         h2[i_obs] = hidden

#         return h2, loss, losses_pre

# class Discretized_GRU(torch.nn.Module):
#     ## Discretized GRU model (GRU-ODE-Bayes without ODE and without Bayes)
#     def __init__(self, input_size, hidden_size, p_hidden, prep_hidden, bias=True, cov_size=1, cov_hidden=1, classification_hidden=1, logvar=True, mixing=1, dropout_rate=0, impute=True):
#         """
#         The smoother variable computes the classification loss as a weighted average of the projection of the latents at each observation.
#         """
        
#         super().__init__()
#         self.impute = impute
#         self.p_model = torch.nn.Sequential(
#             torch.nn.Linear(hidden_size, p_hidden, bias=bias),
#             torch.nn.ReLU(),
#             torch.nn.Dropout(p=dropout_rate),
#             torch.nn.Linear(p_hidden, 2 * input_size, bias=bias),
#         )

#         self.classification_model = torch.nn.Sequential(
#             torch.nn.Linear(hidden_size,classification_hidden,bias=bias),
#             torch.nn.ReLU(),
#             torch.nn.Dropout(p=dropout_rate),
#             torch.nn.Linear(classification_hidden,1,bias=bias)
#         )

#         self.gru = SpatialGRUCell(2*input_size, hidden_size, bias = bias)

#         # if logvar:
#         #     self.gru_obs = GRUObservationCellLogvar(input_size, hidden_size, prep_hidden, bias=bias)
#         # else:
#         self.gru_obs = GRUObservationCell(input_size, hidden_size, prep_hidden, bias=bias)

#         self.covariates_map = torch.nn.Sequential(
#             torch.nn.Linear(cov_size, cov_hidden, bias=bias),
#             torch.nn.ReLU(),
#             torch.nn.Dropout(p=dropout_rate),
#             torch.nn.Linear(cov_hidden, hidden_size, bias=bias),
#             torch.nn.Tanh()
#         )


#         self.input_size = input_size
#         self.logvar     = logvar
#         self.mixing     = mixing #mixing hyperparameter for loss_1 and loss_2 aggregation.

#         self.apply(init_weights)

#     def ode_step(self, state, p, delta_t):
 
#         state = self.gru(p,state)

#         raise ValueError(f"Unknown solver '{self.solver}'.")

#     def forward(self, times, time_ptr, X, M, obs_idx, delta_t, T, cov,
#                 return_path=False, smoother = False, class_criterion = None, labels=None):
#         """
#         Args:
#             times      np vector of observation times
#             time_ptr   start indices of data for a given time
#             X          data tensor
#             M          mask tensor (1.0 if observed, 0.0 if unobserved)
#             obs_idx    observed patients of each datapoint (indexed within the current minibatch)
#             delta_t    time step for Euler
#             T          total time
#             cov        static covariates for learning the first h0
#             return_path   whether to return the path of state

#         Returns:
#             state          hidden state at final time (T)
#             loss       loss of the Gaussian observations
#         """


#         state = self.covariates_map(cov)

#         p            = self.p_model(state)
#         current_time = 0.0
#         counter      = 0

#         loss_1 = 0 #Pre-jump loss
#         loss_2 = 0 #Post-jump loss (KL between p_updated and the actual sample)

#         if return_path:
#             path_t = [0]
#             path_p = [p]
#             path_h = [state]

#         if smoother:
#             class_loss_vec = torch.zeros(cov.shape[0],device = state.device)
#             num_evals_vec  = torch.zeros(cov.shape[0],device = state.device)
#             class_criterion = class_criterion
#             assert class_criterion is not None

#         assert len(times) + 1 == len(time_ptr)
#         assert (len(times) == 0) or (times[-1] <= T)

#         for i, obs_time in enumerate(times):
#             ## Propagation of the ODE until next observation
#             while current_time < (obs_time-0.001*delta_t): #0.0001 delta_t used for numerical consistency.
                
#                 if self.impute is False:
#                     p = torch.zeros_like(p)
#                 state = self.gru(p, state)
#                 p = self.p_model(state)

#                 ## using counter to avoid numerical errors
#                 counter += 1
#                 current_time = counter * delta_t
#                 #Storing the predictions.
#                 if return_path:
#                     path_t.append(current_time)
#                     path_p.append(p)
#                     path_h.append(state)

#             ## Reached an observation
#             start = time_ptr[i]
#             end   = time_ptr[i+1]

#             X_obs = X[start:end]
#             M_obs = M[start:end]
#             i_obs = obs_idx[start:end]

#             ## Using GRUObservationCell to update state. Also updating p and loss
#             state, losses = self.gru_obs(state, p, X_obs, M_obs, i_obs)
           
#             if smoother:
#                 class_loss_vec[i_obs] += class_criterion(self.classification_model(state[i_obs]),labels[i_obs]).squeeze(1)
#                 num_evals_vec[i_obs] +=1
#             loss_1    = loss_1 + losses.sum()
#             p         = self.p_model(state)

#             loss_2 = loss_2 + compute_KL_loss(p_obs = p[i_obs], X_obs = X_obs, M_obs = M_obs, logvar=self.logvar)

#             if return_path:
#                 path_t.append(obs_time)
#                 path_p.append(p)
#                 path_h.append(state)


#         ## after every observation has been processed, propagating until T
#         while current_time < T:
#             if self.impute is False:
#                 p = torch.zeros_like(p)
#             state = self.gru(p,state)
#             p = self.p_model(state)

#             counter += 1
#             current_time = counter * delta_t
#             #Storing the predictions
#             if return_path:
#                 path_t.append(current_time)
#                 path_p.append(p)
#                 path_h.append(state)

#         loss = loss_1 + self.mixing * loss_2

#         if smoother:
#             class_loss_vec += class_criterion(self.classification_model(state),labels).squeeze(1)
#             class_loss_vec /= num_evals_vec
        
#         class_pred = self.classification_model(state)
       
#         if return_path:
#             if smoother:
#                 return state, loss, class_pred, np.array(path_t), torch.stack(path_p), torch.stack(path_h), class_loss_vec
#             else:
#                 return state, loss, class_pred, np.array(path_t), torch.stack(path_p), torch.stack(path_h)
#         else:
#             if smoother:
#                 return state, loss, class_pred, class_loss_vec
#             else:
#                 return state, loss, class_pred
