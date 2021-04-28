from six.moves import urllib
import torch
from src.download.dataloader_mnist import *
from src.download.dataloader_dSprites import *
from src.download.dataloader_OCT import *
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
    datamodule_mnist = MNIST_DataModule(data_dir=data_path / "mnist/",
                                    batch_size=hparams.batch_size,
                                    seed=hparams.seed)

    datamodule_mnist.prepare_data()
    datamodule_mnist.setup()

# dSprites

if hparams.dataset == "dSprites":
    datamodule_dSprites = dSprites_DataModule(data_dir=data_path / "dSprites/",
                                                        batch_size=hparams.batch_size,
                                                        seed=hparams.seed)

    datamodule_dSprites.prepare_data()
    datamodule_dSprites.setup()

if hparams.dataset == "OCT":
    
    system = platform.system()
    if system == "Windows":
        data_path = "C:/Users/Lukas/Documents/GitHub/Semi-supervised-methods/data/OCT"
    else:
        data_path = "/cluster/scratch/luklein/CellData/OCT"

    datamodule_OCT = OCT_DataModule(data_dir=data_path,
                                    batch_size=hparams.batch_size,
                                    seed=hparams.seed)

    datamodule_OCT.setup()
