import torch
from torch import nn
import torch.nn.functional as F
from torch import optim, Tensor
import math
import torchvision

import pytorch_lightning as pl


class betaTCVAE_ResNet(pl.LightningModule):
    def __init__(self,
                 latent_dim=10,
                 input_height=784,
                 lr=0.001,
                 anneal_steps: int = 200,
                 alpha: float = 1.,
                 beta: float = 1.,
                 gamma: float = 1.,
                 M_N=0.005,
                 dataset = "mnist",
                 c=64,
                 pretrained = False
                 ):
        super(betaTCVAE_ResNet, self).__init__()
        self.save_hyperparameters()
        self.c = c
        self.num_iter = 0

        if dataset == "mnist" or dataset == "mnist_small":
            self.scale = 7
            self.trainset_size = 50000
        elif dataset=="dSprites_small":
            self.scale = 16
            self.trainset_size = 600000
        elif dataset == "OCT_small":
            self.scale = 0.25 * 200
            self.trainset_size = 107000

        # Encoder
        self.resnet = torchvision.models.resnet18(pretrained=True)

        if pretrained == True:

            with torch.no_grad():
                weight = self.resnet.conv1.weight.clone()

            self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)

            for param in self.resnet.parameters():
                param.requires_grad = False

            self.resnet.conv1.weight[:, 0] = weight[:, 0]
        else:
            self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
            
        self.resnet.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.enc_fc = nn.Linear(in_features=1000, out_features=5*c)
        self.fc_mu = nn.Linear(in_features=5*c, out_features=latent_dim)
        self.fc_logvar = nn.Linear(in_features=5*c, out_features=latent_dim)

        # Decoder
        self.dec_bn1 = nn.BatchNorm1d(latent_dim)
        self.fc = nn.Linear(in_features=latent_dim,
                            out_features=c*2*self.scale*self.scale)
        self.dec_bn2 = nn.BatchNorm1d(c*2*self.scale*self.scale)
        self.dec_conv2 = nn.ConvTranspose2d(
            in_channels=c*2, out_channels=c, kernel_size=4, stride=2, padding=1)
        self.dec_bn3 = nn.BatchNorm2d(c)
        self.dec_conv1 = nn.ConvTranspose2d(
            in_channels=c, out_channels=1, kernel_size=4, stride=2, padding=1)


    def encode(self, x):
        x = self.resnet(x)
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

    def log_density_gaussian(self, x: Tensor, mu: Tensor, logvar: Tensor):
        norm = - 0.5 * (math.log(2 * math.pi) + logvar)
        log_density = norm - 0.5 * ((x - mu) ** 2 * torch.exp(-logvar))
        return log_density

    def loss(self, recons, x, mu, log_var, z):

        recons_loss = F.binary_cross_entropy(
            recons.view(-1, self.hparams.input_height),
            x.view(-1, self.hparams.input_height),
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
            anneal_rate = min(0 + 1 * self.num_iter /
                              self.hparams.anneal_steps, 1)
        else:
            anneal_rate = 1

        loss = recons_loss/batch_size + \
            self.hparams.alpha * mi_loss + \
            self.hparams.beta * tc_loss + \
            self.hparams.gamma * anneal_rate * kld_loss

        return loss

    def training_step(self, batch, batch_idx):
        x, _ = batch

        mu, log_var = self.encode(x)
        z = self.sampling(mu, log_var)

        recons = self.decode(z)

        vae_loss = self.loss(recons, x, mu, log_var, z)

        self.log('loss', vae_loss, on_epoch=False, prog_bar=True, on_step=True)
        return vae_loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch

        mu, log_var = self.encode(x)
        z = self.sampling(mu, log_var)

        recons = self.decode(z)

        vae_loss = self.loss(recons, x, mu, log_var, z)

        self.log('val_loss', vae_loss, on_epoch=True, prog_bar=True)
        return vae_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        return tqdm_dict
