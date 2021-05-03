import torch
from pathlib import Path
from tools.argparser import *
import pytorch_lightning as pl
import os
import shutil
import zipfile
import numpy as np
import urllib.request as request

from torch.utils.data import DataLoader, random_split, TensorDataset, DistributedSampler
from torch import Tensor
import torch
import torchvision
from torchvision import transforms

def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


class OCT_DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_dir, seed):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seed = seed

    def setup(self):

        transform_img = transforms.Compose([
            torchvision.transforms.Resize((202, 202)),
            transforms.Grayscale(num_output_channels=1),
            #transforms.CenterCrop((496,750)),
            transforms.ToTensor(),
        ])

        train = torchvision.datasets.ImageFolder(self.data_dir + "/train",
                                                 transform=transform_img)

        data_enc, self.train_cla = random_split(train, [107000, 1309],  # 108309
                                    generator=torch.Generator().manual_seed(self.seed))

        self.train_enc, self.val_enc = random_split(data_enc, [85600, 21400],  
                                    generator=torch.Generator().manual_seed(self.seed))

        # if torch.cuda.device_count() > 1:
        #     self.sampler_enc = DistributedSampler(self.train_enc)
        #     self.sampler_cla = DistributedSampler(self.train_cla)

        # else:
        #     weights = make_weights_for_balanced_classes(train.imgs, len(train.classes))
        #     weights = torch.DoubleTensor(weights)
        #     weights_enc, weights_cla = random_split(weights, [107000, 1309],  # 108309
        #                         generator=torch.Generator().manual_seed(self.seed))
        #     weights_enc, _ = random_split(weights_enc, [85600, 21400],
        #                                 generator=torch.Generator().manual_seed(self.seed))

        #     self.sampler_enc = torch.utils.data.sampler.WeightedRandomSampler(weights_enc, len(weights_enc))

        #     self.sampler_cla = torch.utils.data.sampler.WeightedRandomSampler(weights_cla, len(weights_cla))

        self.test = torchvision.datasets.ImageFolder(self.data_dir + "/test", transform=transform_img)

    def train_dataloader(self):
        return DataLoader(self.train_enc, batch_size=self.batch_size)

    def train_dataloader_cla(self):
        return DataLoader(self.train_cla, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.val_enc, batch_size=self.batch_size)

    def val_dataloader_cla(self):
        return DataLoader(self.test, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)
