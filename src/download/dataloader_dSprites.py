from pathlib import Path
import os
import shutil
import zipfile
import numpy as np
import urllib.request as request

import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
from torch import Tensor
import pytorch_lightning as pl

from tools.argparser import *

"""
Creating Pytorch Lightning datamodule for automatic download, preparation 
and setting up of the dataloaders for the dSprites dataset.
"""

def collate_fn(batch): # Function to change every batch from list to tensor when loaded
    imgs = torch.stack([item[0].float() for item in batch])
    targets = torch.stack([item[1] for item in batch])
    return imgs, targets

class dSprites_DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_dir, seed, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers

    def prepare_data(self):
        if not os.path.exists(os.path.join(self.data_dir, "imgs.npy")):
            print("Downloading and extracting dSprites retina data...")

            data_url = 'https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true'

            file = os.path.join(
                self.data_dir, "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")

            os.makedirs(self.data_dir, exist_ok=True)

            with request.urlopen(data_url) as response, open(file, 'wb+') as out_file:
                shutil.copyfileobj(response, out_file)

            zip_ref = zipfile.ZipFile(file, 'r')
            zip_ref.extractall(self.data_dir)
            zip_ref.close()

    def setup(self):
        X = np.load(os.path.join(self.data_dir, "imgs.npy"))
        y = np.load(os.path.join(self.data_dir, "latents_classes.npy"))
        y = y[:, 1]
        
        X = np.expand_dims(X, axis=1)

        dSprites_full = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))

        self.val_cla, self.train_cla, data_enc = random_split(dSprites_full, [1280, 6000, 730000],  
                                                    generator=torch.Generator().manual_seed(self.seed))

        self.train_enc, self.val_enc, self.test = random_split(data_enc, [600000, 65000, 65000],
                                                    generator=torch.Generator().manual_seed(self.seed))

    def train_dataloader(self):
        return DataLoader(self.train_enc, batch_size=self.batch_size, 
                          collate_fn=collate_fn, num_workers=self.num_workers)

    def train_dataloader_cla(self):
        return DataLoader(self.train_cla, batch_size=32, 
                          collate_fn=collate_fn, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_enc, batch_size=self.batch_size,
                          collate_fn=collate_fn, num_workers=self.num_workers)

    def val_dataloader_cla(self):
        return DataLoader(self.val_cla, batch_size=64,
                          collate_fn=collate_fn, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size,  
                          collate_fn=collate_fn, num_workers=self.num_workers)
