#### Setup ####

import warnings
import datetime
import os
import glob
import platform
import subprocess
from pathlib import Path

main_dir = os.path.dirname(Path(__file__).resolve().parents[0])
os.chdir(main_dir)

plt = platform.system()

if plt == "Linux": # Change to your ethz account
    command = 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cluster/home/luklein/miniconda3/lib'
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    command = 'export PYTHONPATH="${PYTHONPATH}:/cluster/home/luklein/Semi-supervised-methods"'
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

import pytorch_lightning as pl

warnings.filterwarnings('ignore')

from src.__init__ import *

## WandB Logging ##

import wandb
#wandb.init(project="VAE-LinReg")
from pytorch_lightning.loggers import WandbLogger

## Parser and Seeding ##

parser = get_parser()
hparams = parser.parse_args()

pl.seed_everything(hparams.seed)

## Remove all saved models ##

files = glob.glob(os.path.dirname(Path(os.getcwd(), "models/encoder/VAE_loss/*/test/"))
        )

for f in files:
    os.remove(f) # As administrator

#### Train Encoder ####

model_enc = betaVAE(beta = 1)

trainer = pl.Trainer(min_epochs = 30, callbacks=[early_stop_callback_VAE,checkpoint_callback_VAE], gpus=torch.cuda.device_count())
trainer.fit(model_enc, datamodule_mnist)


#### Train Classifier ####

#wandb_logger = WandbLogger(project='VAE-LinReg', job_type='train')

model_reg = MLP(
    freeze=True, input_dim=10, num_classes=10, learning_rate=0.001)

trainer = pl.Trainer(
    #logger=wandb_logger, 
    callbacks=[early_stop_callback_cla], 
    gpus=torch.cuda.device_count()
)

trainer.fit(model_reg, datamodule_mnist.train_dataloader(), datamodule_mnist.val_dataloader())

trainer.test(model_reg, datamodule_mnist.test_dataloader())

#wandb.finish()

