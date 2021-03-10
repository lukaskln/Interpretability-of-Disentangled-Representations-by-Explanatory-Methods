from pathlib import Path
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics.functional import accuracy
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import softmax
from torch.optim import Adam
from torch.optim.optimizer import Optimizer

from src.models.encoder.VAE_loss.betaVAE import *

class LogisticRegression(pl.LightningModule):
    """
    Logistic regression model
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        bias: bool = True,
        learning_rate: float = 1e-4,
        optimizer: Optimizer = Adam,
        l1_strength: float = 0.0,
        l2_strength: float = 0.0,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.optimizer = optimizer

        self.encoder = betaVAE.load_from_checkpoint("C:/Users/Lukas/Documents/GitHub/Semi-supervised-methods/models/encoder/VAE_loss/Best_VAE.ckpt")
        self.encoder.freeze()
        self.linear = nn.Linear(in_features=self.hparams.input_dim, out_features=self.hparams.num_classes, bias=bias)

    def forward(self, x):
        x = self.encoder(x)
        x = self.linear(x)
        y_hat = softmax(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch

        # flatten any input
        x = x.view(x.size(0), -1)
        y_hat = self(x)

        # PyTorch cross_entropy function combines log_softmax and nll_loss in single function
        loss = F.cross_entropy(y_hat, y, reduction='sum')

        # L1 regularizer
        if self.hparams.l1_strength > 0:
            l1_reg = sum(param.abs().sum() for param in self.parameters())
            loss += self.hparams.l1_strength * l1_reg

        # L2 regularizer
        if self.hparams.l2_strength > 0:
            l2_reg = sum(param.pow(2).sum() for param in self.parameters())
            loss += self.hparams.l2_strength * l2_reg

        loss /= x.size(0)

        tensorboard_logs = {'train_ce_loss': loss}
        progress_bar_metrics = tensorboard_logs
        return {'loss': loss, 'log': tensorboard_logs, 'progress_bar': progress_bar_metrics}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self(x)
        acc = accuracy(y_hat, y)
        return {'val_loss': F.cross_entropy(y_hat, y), 'acc': acc}

    def validation_epoch_end(self, outputs):
        acc = torch.stack([x['acc'] for x in outputs]).mean()
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_ce_loss': val_loss, 'val_acc': acc}
        progress_bar_metrics = tensorboard_logs
        return {'val_loss': val_loss, 'log': tensorboard_logs, 'progress_bar': progress_bar_metrics}

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self(x)
        acc = accuracy(y_hat, y)
        return {'test_loss': F.cross_entropy(y_hat, y), 'acc': acc}

    def test_epoch_end(self, outputs):
        acc = torch.stack([x['acc'] for x in outputs]).mean()
        test_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_ce_loss': test_loss, 'test_acc': acc}
        progress_bar_metrics = tensorboard_logs
        return {'test_loss': test_loss, 'log': tensorboard_logs, 'progress_bar': progress_bar_metrics}

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.hparams.learning_rate)
