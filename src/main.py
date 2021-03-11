#### Setup ####

import warnings
import datetime
import os
import glob
from pathlib import Path

main_dir = os.path.dirname(Path(__file__).resolve().parents[0])
os.chdir(main_dir)

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

files = glob.glob(
        os.path.join(main_dir, "models\\encoder\\VAE_loss","*")
        )

for f in files:
    os.remove(f) # As administrator

#### Train Encoder ####

model_enc = betaVAE(beta = 4)

trainer = pl.Trainer(callbacks=[early_stop_callback_VAE,checkpoint_callback_VAE], gpus=torch.cuda.device_count())
trainer.fit(model_enc, datamodule_mnist.train_dataloader())


#### Train Classifier ####

#wandb_logger = WandbLogger(project='VAE-LinReg', job_type='train')

model_reg = LogisticRegression(freeze = False, input_dim= 16 , num_classes=10, learning_rate=0.0001)

trainer = pl.Trainer(
    #logger=wandb_logger, 
    callbacks=[early_stop_callback_cla], 
    gpus=torch.cuda.device_count()
)

trainer.fit(model_reg, datamodule_mnist.train_dataloader(), datamodule_mnist.val_dataloader())

trainer.test(model_reg, datamodule_mnist.test_dataloader())

wandb.finish()

