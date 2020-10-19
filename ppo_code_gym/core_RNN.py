import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        # obs_numpy = obs.numpy().tolist()
        # for index in range(5,13):
        #     if obs_numpy[index] == 1:
        #         # print("item is 1")
        #         logits[index-5] = 0
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.



class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(16,8), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        #print('making the actor critics, what is action space n?  ', action_space.n, action_space, flush=True)
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)

        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]


class RNNActor(nn.Module):
     def __init__(self, state_dim, action_dim, continuous_action_space, trainable_std_dev, init_log_std_dev=None):
        super().__init__()
        self.lstm = nn.LSTM(state_dim, hp.hidden_size, num_layers=hp.recurrent_layers)
        self.layer_hidden = nn.Linear(hp.hidden_size, hp.hidden_size)
        self.layer_policy_logits = nn.Linear(hp.hidden_size, action_dim)
        self.action_dim = action_dim
        self.continuous_action_space = continuous_action_space 
        self.log_std_dev = nn.Parameter(init_log_std_dev * torch.ones((action_dim), dtype=torch.float), requires_grad=trainable_std_dev)
        self.covariance_eye = torch.eye(self.action_dim).unsqueeze(0)
        self.hidden_cell = None
        
    def get_init_state(self, batch_size, device):
        self.hidden_cell = (torch.zeros(hp.recurrent_layers, batch_size, hp.hidden_size).to(device),
                            torch.zeros(hp.recurrent_layers, batch_size,hp.hidden_size).to(device))
        
    def forward(self, state, terminal=None):
        batch_size = state.shape[1]
        device = state.device
        if self.hidden_cell is None or batch_size != self.hidden_cell[0].shape[1]:
            self.get_init_state(batch_size, device)
        if terminal is not None:
            self.hidden_cell = [value * (1. - terminal).reshape(1, batch_size, 1) for value in self.hidden_cell]
        _, self.hidden_cell = self.lstm(state, self.hidden_cell)
        hidden_out = F.elu(self.layer_hidden(self.hidden_cell[0][-1]))
        policy_logits_out = self.layer_policy_logits(hidden_out)
        if self.continuous_action_space:
            cov_matrix = self.covariance_eye.to(device).expand(batch_size, self.action_dim, self.action_dim) * torch.exp(self.log_std_dev.to(device))
            # We define the distribution on the CPU since otherwise operations fail with CUDA illegal memory access error.
            policy_dist = torch.distributions.multivariate_normal.MultivariateNormal(policy_logits_out.to("cpu"), cov_matrix.to("cpu"))
        else:
            policy_dist = distributions.Categorical(F.softmax(policy_logits_out, dim=1).to("cpu"))
        return policy_dist

class RNNCritic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.layer_lstm = nn.LSTM(state_dim, hp.hidden_size, num_layers=hp.recurrent_layers)
        self.layer_hidden = nn.Linear(hp.hidden_size, hp.hidden_size)
        self.layer_value = nn.Linear(hp.hidden_size, 1)
        self.hidden_cell = None
        
    def get_init_state(self, batch_size, device):
        self.hidden_cell = (torch.zeros(hp.recurrent_layers, batch_size, hp.hidden_size).to(device),
                            torch.zeros(hp.recurrent_layers, batch_size, hp.hidden_size).to(device))
    
    def forward(self, state, terminal=None):
        batch_size = state.shape[1]
        device = state.device
        if self.hidden_cell is None or batch_size != self.hidden_cell[0].shape[1]:
            self.get_init_state(batch_size, device)
        if terminal is not None:
            self.hidden_cell = [value * (1. - terminal).reshape(1, batch_size, 1) for value in self.hidden_cell]
        _, self.hidden_cell = self.layer_lstm(state, self.hidden_cell)
        hidden_out = F.elu(self.layer_hidden(self.hidden_cell[0][-1]))
        value_out = self.layer_value(hidden_out)
        return value_out