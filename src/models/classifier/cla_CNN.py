from pathlib import Path
import os

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.optimizer import Optimizer

from torchmetrics.functional import confusion_matrix, accuracy
import torchvision

class CNN(pl.LightningModule):
    def __init__(
        self,
        path_ckpt,
        input_dim: int,
        num_classes: int,
        bias: bool = True,
        learning_rate: float = 0.001,
        optimizer: Optimizer = Adam,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.optimizer = optimizer

        model = torchvision.models.vgg16_bn()
        model = list(model.features.children())[1:36]

        self.enc_conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.vgg = nn.Sequential(*model)
        self.enc_avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc_1 = nn.Linear(in_features=512, out_features=256)
        self.fc_2 = nn.Linear(in_features=256, out_features=self.hparams.num_classes)

    def forward(self, x):

        x = self.enc_conv1(x)
        x = self.vgg(x)
        x = self.enc_avgpool(x)

        try:
            x = torch.squeeze(x, dim=3)
            x = torch.squeeze(x, dim=2)
        except IndexError:
            pass

        x = F.relu(self.fc_1(x))
        x = self.fc_2(x)
        y_hat = F.softmax(x, dim=1)

        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)

        loss = F.cross_entropy(y_hat, y, reduction='mean')

        acc = accuracy(y_hat, y)

        self.log('loss', loss, on_epoch=True,
                 sync_dist=True if torch.cuda.device_count() > 1 else False)
        self.log('acc', acc, on_epoch=False,  prog_bar=True,
                 sync_dist=True if torch.cuda.device_count() > 1 else False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

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
