import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import pdb
from reskill.utils.general_utils import AttrDict

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class ActionVAE(nn.Module):
    def __init__(self, n_actions=4, n_obs=43, n_z=10, n_hidden=128, 
                 n_layers=1, n_propr=12, device="cuda"):
        super(ActionVAE, self).__init__()

        self.n_actions = n_actions
        self.n_obs = n_obs #43 #28 
        self.n_hidden = n_hidden 
        self.n_layers = n_layers # number of LSTM layers (stacked)
        self.n_z = n_z
        self.device = device
        self.n_propr = n_propr

        self.criterion = nn.MSELoss(reduction="mean")

        self.e1 = nn.Linear(n_actions + n_obs, 750)
        self.e2 = nn.Linear(750, 750)

        self.mean = nn.Linear(750, self.n_z)
        self.log_std = nn.Linear(750, self.n_z)

        self.d1 = nn.Linear(n_actions + n_obs, 750)
        self.d2 = nn.Linear(750, 750)
        self.d3 = nn.Linear(750, n_actions)

        self.latent_dim = self.n_z
        self.device = device

    def forward(self, x):
        states = x['obs']
        actions = x['actions']
        z = F.relu(self.e1(torch.cat([states, actions], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability 
        log_std = self.log_std(z).clamp(-4, 15)
        q = AttrDict(mu=mean,
                     log_var=log_std)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)

        u = self.decode(states, z)

        return AttrDict(reconstruction=u, q=q, z=z)
    
    def decode(self, state, z=None):
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5,0.5)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))
    
    def vae_loss(self, inputs, output, beta=0.00000001):
        bc_loss = self.criterion(output.reconstruction, inputs["actions"])
        kld_loss = (-0.5 * torch.sum(1 + output.q.log_var - output.q.mu.pow(2) - output.q.log_var.exp())) * beta
        return bc_loss, kld_loss

    def loss(self, inputs, output):
        bc_loss, kld_loss = self.vae_loss(inputs, output)
        total_loss = bc_loss + kld_loss

        return AttrDict(bc_loss=bc_loss,
                    kld_loss=kld_loss,
                    total_loss=total_loss)