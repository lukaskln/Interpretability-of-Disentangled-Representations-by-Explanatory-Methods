import torch
from torch import nn
import torch.nn.functional as F
from torch import optim, Tensor

import pytorch_lightning as pl

class betaVAE(pl.LightningModule):
    def __init__(self,enc_out_dim=256, latent_dim=10, input_height=784, beta = 1):
        super(betaVAE, self).__init__()
        self.save_hyperparameters()
        self.encoder = nn.Sequential(
            nn.Linear(input_height, 500), nn.ReLU(),
            nn.BatchNorm1d(num_features=500),
            nn.Linear(500, 250), nn.ReLU(),
            nn.BatchNorm1d(num_features=250),
            nn.Linear(250, enc_out_dim), nn.ReLU(),
            )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 250), nn.ReLU(),
            nn.BatchNorm1d(num_features=250),
            nn.Linear(250, 500), nn.ReLU(),
            nn.Linear(500, input_height)
            )
        
        self.fc_mu=nn.Linear(enc_out_dim, latent_dim)
        self.fc_log_var=nn.Linear(enc_out_dim, latent_dim)
       
    def encode(self,x):
        z = self.encoder(x)
        mu = self.fc_mu(z)
        log_var = self.fc_log_var(z)
        return mu,log_var
    
    def sampling(self,mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def decode(self, z):
        reconst = self.decoder(z)
        return torch.sigmoid(reconst)

    def forward(self,x):
        mu, log_var = self.encode(x)
        p, q, z = self.sampling(mu, log_var)
        return mu
    

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(-1, self.hparams.input_height)

        mu, log_var = self.encode(x)
        p, q, z = self.sampling(mu, log_var)

        reconst = self.decode(z)

        recon_loss = F.mse_loss(reconst, x, reduction='mean')

        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = log_qz - log_pz
        kl = kl.mean()

        vae_loss = recon_loss + self.hparams.beta*kl

        self.log('loss', vae_loss, on_epoch=False, prog_bar=True, on_step = True)
        return vae_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(-1, self.hparams.input_height)

        mu, log_var = self.encode(x)
        p, q, z = self.sampling(mu, log_var)

        reconst = self.decode(z)

        recon_loss = F.mse_loss(reconst, x, reduction='mean')

        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = log_qz - log_pz
        kl = kl.mean()

        vae_loss = recon_loss + self.hparams.beta*kl

        self.log('val_loss', vae_loss, on_epoch=True, prog_bar=True)
        return vae_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
    
    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        return tqdm_dict
