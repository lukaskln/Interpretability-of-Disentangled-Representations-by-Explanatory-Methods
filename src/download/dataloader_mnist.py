import torch
from pathlib import Path
from tools.argparser import *
import pytorch_lightning as pl
import os

from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms


class MNIST_DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_dir, seed, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)

        data_cla, data_enc = random_split(mnist_full, [1000, 59000],
                                generator=torch.Generator().manual_seed(self.seed))

        self.train_enc, self.val_enc = random_split(data_enc, [50000, 9000],
                                generator=torch.Generator().manual_seed(self.seed))

        self.train_cla, self.val_cla, _ = random_split(data_cla, [400, 100, 500], # 800 ,200
                                generator=torch.Generator().manual_seed(self.seed))

        self.test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_enc, batch_size=self.batch_size, num_workers=self.num_workers)

    def train_dataloader_cla(self):
        return DataLoader(self.train_cla, batch_size=32, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_enc, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader_cla(self):
        return DataLoader(self.val_cla, batch_size=32, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)
