from six.moves import urllib
import torch
from src.download.dataloader_cifar10 import *
from src.download.dataloader_mnist import *
from src.download.dataloader_mnist_small import *
from src.download.dataloader_mnist_small_enc import *
from pathlib import Path
from tools.argparser import *

"""
Downloads training, validation and test data if not available locally and stores them into 
the respective subdirectories.
"""

parser = get_parser()
hparams = parser.parse_args()

### Define data path ###
base_path = Path(__file__).resolve().parents[2]
data_path = base_path / "data"

# Cifar 10
datamodule_cifar10 = CIFAR10DataModule(data_dir = data_path / "cifar-10/", 
                                       batch_size=hparams.batch_size)

datamodule_cifar10.prepare_data()
datamodule_cifar10.setup()

# MNIST
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

datamodule_mnist = MNISTDataModule(data_dir= data_path / "mnist/",
                                   batch_size=hparams.batch_size)

datamodule_mnist.prepare_data()
datamodule_mnist.setup()

datamodule_mnist_small = MNIST_smallDataModule(data_dir=data_path / "mnist/",
                                               batch_size=32,
                                               seed=hparams.seed)

datamodule_mnist_small.prepare_data()
datamodule_mnist_small.setup()

datamodule_mnist_small_enc = MNIST_small_encDataModule(data_dir=data_path / "mnist/",
                                   batch_size=hparams.batch_size,
                                   seed=hparams.seed)

datamodule_mnist_small_enc.prepare_data()
datamodule_mnist_small_enc.setup()
