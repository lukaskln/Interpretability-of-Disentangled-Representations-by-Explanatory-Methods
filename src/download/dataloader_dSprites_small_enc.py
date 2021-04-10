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
from torchvision import transforms


def collate_fn(batch):
    imgs = torch.stack([item[0].float() for item in batch])
    targets = [item[1] for item in batch]
    return imgs, targets

class dSprites_small_encDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_dir, seed):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seed = seed

        self.dims = (1, 64, 64)
        self.num_classes = 3

    def prepare_data(self):
        if not os.path.exists(os.path.join(self.data_dir, "imgs.npy")):
            data_url = 'https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true'

            file = os.path.join(
                self.data_dir, "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")

            os.makedirs(self.data_dir, exist_ok=True)
            with request.urlopen(data_url) as response, open(file, 'wb+') as out_file:
                shutil.copyfileobj(response, out_file)

            zip_ref = zipfile.ZipFile(file, 'r')
            zip_ref.extractall(self.data_dir)
            zip_ref.close()

    def setup(self, stage=None):

        X = np.load(os.path.join(self.data_dir, "imgs.npy"))
        y = np.load(os.path.join(self.data_dir, "latents_classes.npy"))
        y = y[:, 1]
        
        X = np.expand_dims(X, axis=1)

        dSprites_full = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))

        _, data = random_split(dSprites_full, [7280, 730000],  # 737280
                               generator=torch.Generator().manual_seed(self.seed))
        self.dSprites_train, self.dSprites_val, self.dSprites_test = random_split(data, [600000, 100000, 30000],
                                                        generator=torch.Generator().manual_seed(self.seed))

    def train_dataloader(self):
        return DataLoader(self.dSprites_train, batch_size=self.batch_size, shuffle=True,
                          collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.dSprites_val, batch_size=self.batch_size,
                          collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.dSprites_test, batch_size=self.batch_size,  
            collate_fn=collate_fn)
