import torch
from torch import nn
import torch.nn.functional as F
from torch import optim, Tensor

import pytorch_lightning as pl

class betaVAE_CNN(pl.LightningModule):
    def __init__(self, input_height = 784, c=32, latent_dim=10, beta=1, lr=0.001, dataset = "mnist"):
        super(betaVAE_CNN, self).__init__()
        self.c = c
        self.save_hyperparameters()

        if dataset == "mnist" or dataset == "mnist_small":
            self.scale = 7
        elif dataset=="dSprites_small":
            self.scale = 16

        # Encoder
        self.enc_conv1 = nn.Conv2d(in_channels=1, out_channels=c,kernel_size=5, stride=1, padding=1)  # out: c x 14 x 14
        self.enc_pool1 = nn.MaxPool2d(2, stride=2, padding=0)
        self.enc_conv2 = nn.Conv2d(in_channels=c, out_channels=c*2,kernel_size=5, stride=1, padding=1)  # out: c x 7 x 7
        self.enc_pool2 = nn.AdaptiveAvgPool2d((3,3))
        self.enc_fc = nn.Linear(in_features=c*2*3*3, out_features=10*c)
        self.fc_mu = nn.Linear(in_features=10*c, out_features=latent_dim)
        self.fc_logvar = nn.Linear(in_features=10*c, out_features=latent_dim)

        # Decoder
        self.dec_bn1 = nn.BatchNorm1d(latent_dim)
        self.fc = nn.Linear(in_features=latent_dim, out_features=c*2*self.scale*self.scale)
        self.dec_bn2 = nn.BatchNorm1d(c*2*self.scale*self.scale)
        self.dec_conv2 = nn.ConvTranspose2d(in_channels=c*2, out_channels=c, kernel_size=4, stride=2, padding=1)
        self.dec_bn3 = nn.BatchNorm2d(c)
        self.dec_conv1 = nn.ConvTranspose2d(in_channels=c, out_channels=1, kernel_size=4, stride=2, padding=1)

    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x = self.enc_pool1(x)
        x = F.relu(self.enc_conv2(x))
        x = self.enc_pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.enc_fc(x))
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        return mu, log_var

    def sampling(self, mu, log_var):
        std = torch.exp(log_var * 0.5)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return z

    def decode(self, z):
        x = self.dec_bn1(z)
        x = self.fc(x)
        x = self.dec_bn2(x)
        x = x.view(x.size(0), self.c*2, self.scale, self.scale)
        x = F.relu(self.dec_conv2(x))
        x = self.dec_bn3(x)
        x = self.dec_conv1(x)
        return torch.sigmoid(x)

    def forward(self, x):
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

        mu, log_var = self.encode(x)
        z = self.sampling(mu, log_var)

        recons = self.decode(z)

        vae_loss = self.loss(recons, x, mu, log_var)

        self.log('loss', vae_loss, on_epoch=False, prog_bar=True, on_step=True)
        return vae_loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch

        mu, log_var = self.encode(x)
        z = self.sampling(mu, log_var)

        recons = self.decode(z)

        vae_loss = self.loss(recons, x, mu, log_var)

        self.log('val_loss', vae_loss, on_epoch=True, prog_bar=True)
        return vae_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        return tqdm_dict
