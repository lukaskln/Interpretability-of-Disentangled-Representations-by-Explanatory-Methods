from six.moves import urllib
import torch
from src.download.dataloader_cifar10 import *
from src.download.dataloader_mnist import *
from pathlib import Path

"""
Downloads training, validation and test data if not available locally and stores them into 
the respective subdirectories.
"""

### Define data path ###
base_path = Path(__file__).resolve().parents[2]
data_path = base_path / "data"

# Cifar 10
datamodule_cifar10 = CIFAR10DataModule(data_dir = data_path / "cifar-10/" , 
    batch_size=32)

datamodule_cifar10.prepare_data()
datamodule_cifar10.setup()

# MNIST
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

datamodule_mnist = MNISTDataModule(data_dir= data_path / "mnist/",
    batch_size=32)

datamodule_mnist.prepare_data()
datamodule_mnist.setup()
