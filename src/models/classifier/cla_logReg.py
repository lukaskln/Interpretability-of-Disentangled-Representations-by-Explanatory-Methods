from pathlib import Path
import os

import pytorch_lightning as pl
import torch
from torchmetrics.functional import confusion_matrix, accuracy
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.optimizer import Optimizer

from src.models.encoder.VAE_loss.betaVAE import *
from src.models.encoder.VAE_loss.betaVAE_VGG import *
from src.models.encoder.VAE_loss.betaTCVAE import *
from src.models.encoder.VAE_loss.betaTCVAE_VGG import *
from src.models.encoder.VAE_loss.betaVAE_ResNet import *
from src.models.encoder.VAE_loss.betaTCVAE_ResNet import *

class LogisticRegression(pl.LightningModule):
    def __init__(
        self,
        path_ckpt,
        latent_dim: int,
        num_classes: int,
        bias: bool = True,
        learning_rate: float = 0.0001,
        optimizer: Optimizer = Adam,
        l1_strength: float = 0.0,
        l2_strength: float = 0.0,
        VAE_type="betaVAE_MLP",
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.optimizer = optimizer
        self.VAE_type = VAE_type

        if VAE_type == "betaVAE_MLP":
            self.encoder = betaVAE.load_from_checkpoint(path_ckpt)
        elif VAE_type == "betaVAE_VGG":
            self.encoder = betaVAE_VGG.load_from_checkpoint(path_ckpt)
        elif VAE_type == "betaVAE_ResNet":
            self.encoder = betaVAE_ResNet.load_from_checkpoint(path_ckpt)
        elif VAE_type == "betaTCVAE_MLP":
            self.encoder = betaTCVAE.load_from_checkpoint(path_ckpt)
        elif VAE_type == "betaTCVAE_VGG":
            self.encoder = betaTCVAE_VGG.load_from_checkpoint(path_ckpt)
        elif VAE_type == "betaTCVAE_ResNet":
            self.encoder = betaTCVAE_ResNet.load_from_checkpoint(path_ckpt)

        self.encoder.freeze()
            
        self.linear = nn.Linear(in_features=self.hparams.latent_dim, out_features=self.hparams.num_classes, bias=bias)

    def forward(self, x):

        if x.shape[1] != self.hparams.latent_dim:
            x = self.encoder(x)
            
        x = self.linear(x)
        y_hat = F.softmax(x, dim=1)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch

        if self.VAE_type == "betaVAE_MLP" or self.VAE_type == "betaTCVAE_MLP":
            # flatten any input
            x = x.view(x.size(0), -1)

        y_hat = self(x)

        loss = F.cross_entropy(y_hat, y, reduction='mean')

        # L1 regularizer
        if self.hparams.l1_strength > 0:
            l1_reg = sum(param.abs().sum() for param in self.parameters())
            loss += self.hparams.l1_strength * l1_reg

        # L2 regularizer
        if self.hparams.l2_strength > 0:
            l2_reg = sum(param.pow(2).sum() for param in self.parameters())
            loss += self.hparams.l2_strength * l2_reg

        acc = accuracy(y_hat, y)

        self.log('loss', loss, on_epoch=True,
                 sync_dist=True if torch.cuda.device_count() > 1 else False)
        self.log('acc', acc, on_epoch=False,  prog_bar=True,
                 sync_dist=True if torch.cuda.device_count() > 1 else False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        if self.VAE_type == "betaVAE_MLP" or self.VAE_type == "betaTCVAE_MLP":
            x = x.view(x.size(0), -1)

        y_hat = self(x)

        val_loss = F.cross_entropy(y_hat, y, reduction='mean')

        return {'val_loss': val_loss, 'val_y': y, 'val_y_hat': y_hat}

    def validation_epoch_end(self, outputs):
        val_y = torch.cat(tuple([x['val_y'] for x in outputs]))
        val_y_hat = torch.cat(tuple([x['val_y_hat'] for x in outputs]))

        val_loss = F.cross_entropy(val_y_hat, val_y, reduction='mean')
        acc = accuracy(val_y_hat, val_y)

        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True,
                 sync_dist=True if torch.cuda.device_count() > 1 else False)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True,
                 sync_dist=True if torch.cuda.device_count() > 1 else False)

    def test_step(self, batch, batch_idx):
        x, y = batch

        if self.VAE_type == "betaVAE_MLP" or self.VAE_type == "betaTCVAE_MLP":
            x = x.view(x.size(0), -1)
        
        y_hat = self(x)

        test_loss = F.cross_entropy(y_hat, y, reduction='mean')

        return {'test_loss': test_loss, 'test_y': y, 'test_y_hat': y_hat}

    def test_epoch_end(self, outputs):
        test_y = torch.cat(tuple([x['test_y'] for x in outputs]))
        test_y_hat = torch.cat(tuple([x['test_y_hat'] for x in outputs]))

        test_loss = F.cross_entropy(test_y_hat, test_y, reduction='mean')

        acc = accuracy(test_y_hat, test_y)
        confmat = confusion_matrix(test_y_hat, test_y, num_classes=self.hparams.num_classes)

        self.log('test_loss', test_loss, on_epoch=True, prog_bar=True,
                 sync_dist=True if torch.cuda.device_count() > 1 else False)
        self.log('test_acc', acc, on_epoch=True, prog_bar=True,
                 sync_dist=True if torch.cuda.device_count() > 1 else False)

        print("\n Confusion Matrix: \n", torch.round(confmat).type(torch.IntTensor))

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.hparams.learning_rate)

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        return tqdm_dict
