import torch
from pathlib import Path
from tools.argparser import *
import pytorch_lightning as pl
import os
import shutil
import zipfile
import numpy as np
import urllib.request as request

from torch.utils.data import DataLoader, random_split, TensorDataset
from torch import Tensor
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

class OCT_small_DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_dir, seed):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seed = seed

    def setup(self):

        transform_img = transforms.Compose([
            torchvision.transforms.Resize((300, 300)),
            transforms.Grayscale(num_output_channels=1),
            #transforms.CenterCrop((496,750)),
            transforms.ToTensor(),
        ])

        train = torchvision.datasets.ImageFolder(self.data_dir + "/train",
                                                 transform=transform_img)

        self.train_enc, self.train_cla = random_split(train, [107000, 1309],  # 108309
                                                      generator=torch.Generator().manual_seed(self.seed))

        self.test = torchvision.datasets.ImageFolder(self.data_dir + "/test",
                                         transform=transform_img)

    def train_dataloader(self):
        return DataLoader(self.train_enc, batch_size=self.batch_size, shuffle=True)

    def train_cla_dataloader(self):
        return DataLoader(self.train_cla, batch_size=32, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)
