from pathlib import Path
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics.functional import accuracy
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.optimizer import Optimizer

from src.models.encoder.VAE_loss.betaVAE import *
from src.models.encoder.VAE_loss.betaTCVAE import *
from src.models.encoder.VAE_loss.betaVAE_CNN import *
from src.models.encoder.VAE_loss.betaTCVAE_CNN import *

path_ckpt = Path(__file__).resolve().parents[3] / "models/encoder/VAE_loss/Best_VAE.ckpt"


class MLP(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        bias: bool = True,
        learning_rate: float = 1e-4,
        optimizer: Optimizer = Adam,
        VAE_CNN = False,
        TCVAE=False,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.optimizer = optimizer
        self.VAE_CNN = VAE_CNN

        if TCVAE==True:
            if VAE_CNN == False:
                self.encoder = betaTCVAE.load_from_checkpoint(path_ckpt)
            else:
                self.encoder = betaTCVAE_CNN.load_from_checkpoint(path_ckpt)
        else:    
            if VAE_CNN==False:
                self.encoder = betaVAE.load_from_checkpoint(path_ckpt)
            else:
                self.encoder = betaVAE_CNN.load_from_checkpoint(path_ckpt)

        self.encoder.freeze()

        self.fc1 = nn.Linear(self.hparams.input_dim, 256, bias=bias)
        self.fc2 = nn.Linear(256, self.hparams.num_classes, bias=bias)

    def forward(self, x):
        x = self.encoder(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        y_hat = F.softmax(x, dim=1)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch

        if self.VAE_CNN == False:
            # flatten any input
            x = x.view(x.size(0), -1)
        
        y_hat = self(x)

        loss = F.cross_entropy(y_hat, y, reduction='sum')

        loss /= x.size(0)

        acc = accuracy(y_hat, y)

        self.log('loss', loss, on_epoch=True)
        self.log('acc', acc, on_epoch=False,  prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        if self.VAE_CNN == False:
            x = x.view(x.size(0), -1)

        y_hat = self(x)

        acc = accuracy(y_hat, y)
        val_loss = F.cross_entropy(y_hat, y)

        return {'val_loss': val_loss, 'acc': acc}

    def validation_epoch_end(self, outputs):
        acc = torch.stack([x['acc'] for x in outputs]).mean()
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch

        if self.VAE_CNN == False:
            x = x.view(x.size(0), -1)
        
        y_hat = self(x)

        acc = accuracy(y_hat, y)
        test_loss = F.cross_entropy(y_hat, y)

        return {'test_loss': test_loss, 'acc': acc}

    def test_epoch_end(self, outputs):
        acc = torch.stack([x['acc'] for x in outputs]).mean()
        test_loss = torch.stack([x['test_loss'] for x in outputs]).mean()

        self.log('test_loss', test_loss, on_epoch=True, prog_bar=True)
        self.log('test_acc', acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.hparams.learning_rate)

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        return tqdm_dict
