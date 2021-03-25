#### Setup ####
import warnings
import datetime
import os
import glob
from pathlib import Path

main_dir = os.path.dirname(Path(__file__).resolve().parents[0])
os.chdir(main_dir)

import pytorch_lightning as pl
import torch

warnings.filterwarnings('ignore')

print("Loading Modules...")

from src.__init__ import *

from src.evaluation.vis_latentspace import *
from src.evaluation.vis_reconstructions import *

parser = get_parser()
hparams = parser.parse_args()

## Model import ##

path_ckpt = Path(os.getcwd(), 
                "models/encoder/VAE_loss/", 
                ("VAE_in_use_" + str(hparams.eval_model_ID) + ".ckpt")
            )

if os.path.exists(path_ckpt)==False:
    print("[ERROR] Model does not exist. Please check the /models folder for existing models.")
    raise SystemExit(0)

architectures = [
    betaVAE,
    betaVAE_CNN,
    betaVAE_ResNet,
    betaTCVAE,
    betaTCVAE_CNN,
    betaTCVAE_ResNet
]

info = None

for architecture in architectures:
    try:
        encoder = architecture.load_from_checkpoint(path_ckpt)
        encoder_type = architecture.__name__
        break
    except AttributeError:
        # repeat the loop on failure
        continue

## Dataset selection ##
print("Loading Datasets...")

if hparams.dataset == "mnist":
    datamodule_enc = datamodule_mnist
    dataloader_train = datamodule_mnist.train_dataloader()
    dataloader_val = datamodule_mnist.val_dataloader()
    dataloader_test = datamodule_mnist.test_dataloader()
    input_height = 784
    num_classes = 10
elif hparams.dataset == "mnist_small":
    datamodule_enc = datamodule_mnist_small_enc
    dataloader_train = datamodule_mnist_small.train_dataloader()
    dataloader_val = datamodule_mnist_small.val_dataloader()
    dataloader_test = datamodule_mnist_small.test_dataloader()
    input_height = 784
    num_classes = 10
elif hparams.dataset == "dSprites_small":
    datamodule_enc = datamodule_dSprites_small_enc
    dataloader_train = datamodule_dSprites_small.train_dataloader()
    dataloader_val = datamodule_dSprites_small.val_dataloader()
    dataloader_test = datamodule_dSprites_small.test_dataloader()
    input_height = 4096
    num_classes = 6

#### Visualizations ####

try:
    vis_Reconstructions(encoder, dataloader_test, type=encoder_type).visualise()
except RuntimeError:
    print("[ERROR] Wrong dataset selected? Check --dataset=...")
    raise SystemExit(0)


vis_LatentSpace(encoder,
                latent_dim=encoder.state_dict()['fc_mu.weight'].shape[0],
                latent_range=hparams.eval_latent_range
                ).visualise()

#### Disentanglement scores ####

#### Attribution Methods ####


