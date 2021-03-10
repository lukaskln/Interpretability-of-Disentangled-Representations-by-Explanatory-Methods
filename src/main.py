#### Setup ####

import datetime
import os
import glob
from pathlib import Path

os.chdir(os.path.dirname(Path(__file__).resolve().parents[0]))

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from src.__init__ import *

import wandb
#wandb.init(project="VAE-LinReg")
from pytorch_lightning.loggers import WandbLogger

import warnings
warnings.filterwarnings('ignore')

parser = get_parser()
hparams = parser.parse_args()

pl.seed_everything(hparams.seed)

early_stop_callback_VAE = EarlyStopping(
    monitor='vae_loss',
    min_delta=0.001,
    patience=10,
    verbose=False,
    mode='max'
)

checkpoint_callback_VAE = ModelCheckpoint(
    dirpath=os.path.dirname(Path(os.getcwd(), "models/encoder/VAE_loss/test/")),
    save_top_k=1,
    verbose=True,
    monitor='vae_loss',
    mode='min',
    prefix='',
    filename = "Best_VAE"
)

files = glob.glob(os.path.dirname(Path(os.getcwd(), "models/encoder/VAE_loss/*/test")))
for f in files:
    os.remove(f) # As administrator

early_stop_callback_cla = EarlyStopping(
    monitor='loss',
    min_delta=0.001,
    patience=10,
    verbose=False,
    mode='max'
)
#### Train Encoder ####

model_enc = betaVAE()

trainer = pl.Trainer(callbacks=[early_stop_callback_VAE,checkpoint_callback_VAE], gpus=torch.cuda.device_count())
trainer.fit(model_enc, datamodule_mnist)


#### Train Classifier ####

wandb_logger = WandbLogger(project='VAE-LinReg', job_type='train')

model_reg = LogisticRegression(input_dim= 16 , num_classes=10, learning_rate=0.001)

trainer = pl.Trainer(logger=wandb_logger, callbacks=[early_stop_callback_cla], gpus=torch.cuda.device_count())

trainer.fit(model_reg, datamodule_mnist.train_dataloader(), datamodule_mnist.val_dataloader())

trainer.test(model_reg, datamodule_mnist.test_dataloader())

wandb.finish()

