from six.moves import urllib
import torch
from src.download.dataloader_mnist import *
from src.download.dataloader_mnist_small import *
from src.download.dataloader_mnist_small_enc import *
from src.download.dataloader_dSprites_small import *
from src.download.dataloader_dSprites_small_enc import *
from src.download.dataloader_OCT_small import *
from pathlib import Path
from tools.argparser import *
import platform

"""
Downloads training, validation and test data if not available locally and stores them into 
the respective subdirectories.
"""

parser = get_parser()
hparams = parser.parse_args()

### Define data path ###
base_path = Path(__file__).resolve().parents[2]
data_path = base_path / "data"

opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

# MNIST
if hparams.dataset=="mnist":
    datamodule_mnist = MNISTDataModule(data_dir= data_path / "mnist/",
                                    batch_size=hparams.batch_size)

    datamodule_mnist.prepare_data()
    datamodule_mnist.setup()

if hparams.dataset == "mnist_small":
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

# dSprites

if hparams.dataset == "dSprites_small":
    datamodule_dSprites_small_enc = dSprites_small_encDataModule(data_dir=data_path / "dSprites/",
                                                        batch_size=hparams.batch_size,
                                                        seed=hparams.seed)

    datamodule_dSprites_small_enc.prepare_data()
    datamodule_dSprites_small_enc.setup()

    datamodule_dSprites_small = dSprites_smallDataModule(data_dir=data_path / "dSprites/",
                                                                batch_size=32,
                                                                seed=hparams.seed)

    datamodule_dSprites_small.prepare_data()
    datamodule_dSprites_small.setup()

if hparams.dataset == "OCT_small":
    
    system = platform.system()
    if system == "Windows":
        data_path = "C:/Users/Lukas/Documents/GitHub/Semi-supervised-methods/data/OCT"
    else:
        data_path = "/cluster/scratch/luklein/CellData/OCT"

    datamodule_OCT_small = OCT_small_DataModule(data_dir=data_path,
                                                                 batch_size=hparams.batch_size,
                                                                 seed=hparams.seed)

    datamodule_OCT_small.setup()
