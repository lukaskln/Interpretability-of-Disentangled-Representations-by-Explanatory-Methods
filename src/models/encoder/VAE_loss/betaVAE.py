import torch
from torch import nn
import torch.nn.functional as F
from torch import optim, Tensor

import pytorch_lightning as pl

class betaVAE(pl.LightningModule):
    def __init__(self, enc_out_dim=256, latent_dim=10, input_height=784, beta=1, lr=0.001):
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
        return mu, log_var
    
    def sampling(self,mu, log_var):
        std = torch.exp(log_var * 0.5)
        q = torch.distributions.Normal(mu + 1e-05, std + 1e-05)
        z = q.rsample()
        return z

    def decode(self, z):
        reconst = F.sigmoid(self.decoder(z))
        return reconst

    def forward(self,x):
        # During inference, we simply spit out the mean of the
        # learned distribution for the current input.  We could
        # use a random sample from the distribution, but mu of
        # course has the highest probability (MAP Estimator).
        mu, log_var = self.encode(x)
        return mu

    def loss(self, recons, x, mu, logvar):
        bce = F.binary_cross_entropy(
            recons.view(-1, self.hparams.input_height), 
            x.view(-1, self.hparams.input_height),
            reduction='sum')

        batch_size = x.shape[0]

        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (bce + self.hparams.beta*kld)/batch_size
    

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(-1, self.hparams.input_height)
        mu, log_var = self.encode(x)
        z = self.sampling(mu, log_var)

        recons = self.decode(z)

        vae_loss = self.loss(recons, x, mu, log_var)

        self.log('loss', vae_loss, on_epoch=False, prog_bar=True, on_step = True)
        return vae_loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(-1, self.hparams.input_height)
        mu, log_var = self.encode(x)
        z = self.sampling(mu, log_var)

        recons = self.decode(z)

        vae_loss = self.loss(recons, x, mu, log_var)

        self.log('val_loss', vae_loss, on_epoch=True, prog_bar=True)
        return vae_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr= self.hparams.lr)
    
    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        return tqdm_dict
