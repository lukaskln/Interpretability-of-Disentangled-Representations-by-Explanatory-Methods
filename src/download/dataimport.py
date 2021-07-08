from six.moves import urllib
from pathlib import Path

from src.download.dataloader_mnist import *
from src.download.dataloader_dSprites import *
from src.download.dataloader_OCT import *
from tools.argparser import *

"""
Calls the download of training, validation and test data if not available locally and stores them into 
the respective subdirectories. Calls the selected dataloader.
"""

parser = get_parser()
hparams = parser.parse_args()

#### Define data path ####
base_path = Path(__file__).resolve().parents[2]
data_path = base_path / "data"

# Set browser to Mozilla 5.0 in user agent, 
# since chromium does not always work for mnist download:

opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

# MNIST
if hparams.dataset=="mnist":
    datamodule_mnist = MNIST_DataModule(data_dir=data_path / "mnist/",
                                    batch_size=hparams.batch_size,
                                    seed=hparams.seed,
                                    num_workers = hparams.num_workers)

    datamodule_mnist.prepare_data()
    datamodule_mnist.setup()

# dSprites
if hparams.dataset == "dSprites":
    datamodule_dSprites = dSprites_DataModule(data_dir=data_path / "dSprites/",
                                                        batch_size=hparams.batch_size,
                                                        seed=hparams.seed,
                                                        num_workers = hparams.num_workers)

    datamodule_dSprites.prepare_data()
    datamodule_dSprites.setup()

# OCT
if hparams.dataset == "OCT":
    datamodule_OCT = OCT_DataModule(data_dir=data_path / "OCT/",
                                    batch_size=hparams.batch_size,
                                    seed=hparams.seed,
                                    num_workers=hparams.num_workers)

    datamodule_OCT.prepare_data()
    datamodule_OCT.setup()
