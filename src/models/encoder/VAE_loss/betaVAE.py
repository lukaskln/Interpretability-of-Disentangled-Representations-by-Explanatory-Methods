import torch
from torch import nn
import torch.nn.functional as F
from torch import optim, Tensor

import pytorch_lightning as pl

class betaVAE(pl.LightningModule):
    def __init__(self,enc_out_dim=256, latent_dim=20, input_height=784, beta = 1):
        super(betaVAE, self).__init__()
        self.save_hyperparameters()
        self.encoder = nn.Sequential(
            nn.Linear(input_height, 500), nn.ReLU(),
            nn.BatchNorm1d(num_features=500),
            nn.Linear(500, 250), nn.ReLU(),
            nn.BatchNorm1d(num_features=250),
            nn.Linear(250, 50), nn.ReLU(),
            nn.BatchNorm1d(num_features=50),
            nn.Linear(50, enc_out_dim), nn.ReLU(),
            )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 50), nn.ReLU(),
            nn.BatchNorm1d(num_features=50),
            nn.Linear(50, 250), nn.ReLU(),
            nn.BatchNorm1d(num_features=250),
            nn.Linear(250, 500), nn.ReLU(),
            nn.BatchNorm1d(num_features=500),
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
        std = torch.exp(log_var * 0.5) + 0.00001
        q = torch.distributions.Normal(mu + 0.00001, std)
        z = q.rsample()
        return z

    def forward(self,x):
        mu, log_var = self.encode(x)
        z = self.sampling(mu, log_var)
        return z

    def decode(self,z):
        reconst=self.decoder(z)
        return torch.sigmoid(reconst)       

    def loss(self,recons,x, mu, logvar):
        bce = F.binary_cross_entropy(recons, x)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return bce + self.hparams.beta * kld

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x= x.view(-1,self.hparams.input_height)
        mu,log_var = self.encode(x)
        z = self.sampling(mu, log_var)
 
        recons = self.decode(z)

        vae_loss=self.loss(recons,x,mu,log_var)
        self.log_dict({'vae_loss': vae_loss.mean()})
        return vae_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
    
    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        return tqdm_dict
