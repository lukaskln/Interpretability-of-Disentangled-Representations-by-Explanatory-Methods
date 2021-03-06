import math

import torch
import torch.nn.functional as F
from torch import optim, Tensor
from torch import nn

import torchvision
import pytorch_lightning as pl

"""
Defines the MLP encoder with beta-TCVAE loss module. Also for the decoder
an MLP architecture is selected.
"""

class betaTCVAE(pl.LightningModule):
    def __init__(self, 
        latent_dim=10, 
        input_dim=784,
        lr=0.001,
        anneal_steps: int = 200,
        alpha: float = 1.,
        beta: float = 4.,
        gamma: float = 1.,
        dataset="mnist"
        ):
        super(betaTCVAE, self).__init__()
        self.save_hyperparameters()
        self.num_iter = 0

        if dataset == "mnist":
            self.trainset_size = 50000
        elif dataset == "dSprites":
            self.trainset_size = 600000
        elif dataset == "OCT":
            self.trainset_size = 85600

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500), nn.ReLU(),
            nn.BatchNorm1d(num_features=500),
            nn.Linear(500, 250), nn.ReLU(),
            nn.BatchNorm1d(num_features=250),
            nn.Linear(250, 50), nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 250), nn.ReLU(),
            nn.BatchNorm1d(num_features=250),
            nn.Linear(250, 500), nn.ReLU(),
            nn.Linear(500, input_dim)
        )

        self.fc_mu = nn.Linear(50, latent_dim)
        self.fc_log_var = nn.Linear(50, latent_dim)

    def encode(self, x):
        z = self.encoder(x)
        mu = self.fc_mu(z)
        log_var = self.fc_log_var(z)
        return mu, log_var

    def sampling(self, mu, log_var):
        std = torch.exp(log_var * 0.5)
        q = torch.distributions.Normal(mu + 1e-05, std + 1e-05)
        z = q.rsample()
        return z

    def decode(self, z):
        reconst = F.sigmoid(self.decoder(z))
        return reconst

    def forward(self, x):
        mu, log_var = self.encode(x)
        return mu

    def log_density_gaussian(self, x: Tensor, mu: Tensor, logvar: Tensor):
        norm = - 0.5 * (math.log(2 * math.pi) + logvar)
        log_density = norm - 0.5 * ((x - mu) ** 2 * torch.exp(-logvar))
        return log_density

    def loss(self, recons, x, mu, log_var, z):
        # Inspired by: https://github.com/YannDubs/disentangling-vae/blob/7b8285baa19d591cf34c652049884aca5d8acbca/disvae/models/losses.py#L316
        recons_loss = F.binary_cross_entropy(
            recons.view(-1, self.hparams.input_dim),
            x.view(-1, self.hparams.input_dim),
            reduction='sum')

        log_q_zx = self.log_density_gaussian(z, mu, log_var).sum(dim=1)

        zeros = torch.zeros_like(z)
        log_p_z = self.log_density_gaussian(z, zeros, zeros).sum(dim=1)

        batch_size, latent_dim = z.shape
        mat_log_q_z = self.log_density_gaussian(z.view(batch_size, 1, latent_dim),
                                                mu.view(1, batch_size,
                                                        latent_dim),
                                                log_var.view(1, batch_size, latent_dim))
        
        # Estimate the three KL terms (log(q(z))) via importance sampling
        strat_weight = (self.trainset_size - batch_size + 1) / \
            (self.trainset_size * (batch_size - 1))
        
        importance_weights = torch.Tensor(batch_size, batch_size).fill_(
            1 / (batch_size - 1)).to(x.device)
        
        importance_weights.view(-1)[::batch_size] = 1 / self.trainset_size
        importance_weights.view(-1)[1::batch_size] = strat_weight
        importance_weights[batch_size - 2, 0] = strat_weight
        log_importance_weights = importance_weights.log()

        mat_log_q_z += log_importance_weights.view(batch_size, batch_size, 1)

        log_q_z = torch.logsumexp(mat_log_q_z.sum(2), dim=1, keepdim=False)
        log_prod_q_z = torch.logsumexp(
            mat_log_q_z, dim=1, keepdim=False).sum(1)


        # Three KL Term components
        mi_loss = (log_q_zx - log_q_z).mean()
        tc_loss = (log_q_z - log_prod_q_z).mean()
        kld_loss = (log_prod_q_z - log_p_z).mean()

        if self.training:
            self.num_iter += 1
            anneal_rate = min(0 + 1 * self.num_iter / self.hparams.anneal_steps, 1)
        else:
            anneal_rate = 1

        loss = recons_loss/batch_size + \
            self.hparams.alpha * mi_loss + \
            self.hparams.beta * tc_loss + \
            self.hparams.gamma * anneal_rate * kld_loss

        return loss

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(-1, self.hparams.input_dim)
        mu, log_var = self.encode(x)
        z = self.sampling(mu, log_var)

        recons = self.decode(z)

        vae_loss = self.loss(recons, x, mu, log_var, z)

        self.log('loss', vae_loss, on_epoch=False, prog_bar=True, on_step=True,
                 sync_dist=True if torch.cuda.device_count() > 1 else False)
                 
        return vae_loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(-1, self.hparams.input_dim)
        mu, log_var = self.encode(x)
        z = self.sampling(mu, log_var)

        recons = self.decode(z)

        vae_loss = self.loss(recons, x, mu, log_var, z)

        self.log('val_loss', vae_loss, on_epoch=True, prog_bar=True,
                 sync_dist=True if torch.cuda.device_count() > 1 else False)

        return vae_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        return tqdm_dict
