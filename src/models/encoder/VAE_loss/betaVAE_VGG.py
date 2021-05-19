import torch
from torch import nn
import torch.nn.functional as F
from torch import optim, Tensor
import torchvision

import pytorch_lightning as pl

class betaVAE_VGG(pl.LightningModule):
    def __init__(self, input_height = 784, c=64, latent_dim=10, beta=1, lr=0.001, dataset = "mnist"):
        super(betaVAE_VGG, self).__init__()
        self.save_hyperparameters()
        self.c = c
        self.num_iter = 0
        self.dataset = dataset

        if dataset == "mnist":
            self.scale = 7
        elif dataset=="dSprites":
            self.scale = 16
        elif dataset == "OCT":
            self.scale = 50

        # Encoder
        model = torchvision.models.vgg19_bn()

        model = list(model.features.children())[1:53]

        self.enc_conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.vgg = nn.Sequential(*model)
        self.enc_avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.enc_fc = nn.Linear(in_features=512, out_features=10*c)
        self.fc_mu = nn.Linear(in_features=10*c, out_features=latent_dim)
        self.fc_logvar = nn.Linear(in_features=10*c, out_features=latent_dim) 

        # Decoder

        if dataset == "OCT":
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(in_channels=latent_dim, out_channels=self.scale * 8, kernel_size=12, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.scale * 8),
                nn.ReLU(True),
                nn.ConvTranspose2d(in_channels=self.scale * 8, out_channels=self.scale * 4, kernel_size=12, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(self.scale * 4),
                nn.ReLU(True),
                nn.ConvTranspose2d(in_channels=self.scale * 4, out_channels=self.scale * 2, kernel_size=14, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(self.scale * 2),
                nn.ReLU(True),
                nn.ConvTranspose2d(in_channels=self.scale * 2, out_channels=self.scale, kernel_size=14, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(self.scale),
                nn.ReLU(True),
                nn.ConvTranspose2d(in_channels=self.scale, out_channels=1, kernel_size=14, stride=2, padding=1, bias=False),
                nn.Sigmoid()
            )
        else:
            self.decoder = nn.Sequential(
                nn.BatchNorm1d(latent_dim),
                nn.Linear(in_features=latent_dim, out_features=c*2*self.scale*self.scale),
                nn.BatchNorm1d(c*2*self.scale*self.scale),
                nn.Unflatten(1, (self.c*2, self.scale, self.scale)),
                nn.ConvTranspose2d(in_channels=c*2, out_channels=c, kernel_size=4, stride=2, padding=1),
                nn.ReLU(), nn.BatchNorm2d(c),
                nn.ConvTranspose2d(in_channels=c, out_channels=1, kernel_size=4, stride=2, padding=1),
                nn.Sigmoid()
            )

    def encode(self, x):
        x = self.enc_conv1(x)
        x = self.vgg(x)
        x = self.enc_avgpool(x)

        try:
            x = torch.squeeze(x, dim=3)
            x = torch.squeeze(x, dim=2)
        except IndexError:
            pass

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
        if self.dataset=="OCT":
            z = torch.unsqueeze(z,2)
            z = torch.unsqueeze(z,3)

        x = self.decoder(z)
        return x

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

        self.log('loss', vae_loss, on_epoch=False, prog_bar=True, on_step=True,
                 sync_dist=True if torch.cuda.device_count() > 1 else False)
        return vae_loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch

        mu, log_var = self.encode(x)
        z = self.sampling(mu, log_var)

        recons = self.decode(z)

        vae_loss = self.loss(recons, x, mu, log_var)

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
