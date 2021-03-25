import os
from pathlib import Path
from random import randint

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from tools.argparser import *
parser = get_parser()
hparams = parser.parse_args()

model_ID = "VAE_in_use_" + str(randint(1000,9999))

#### VAE Callbacks ####

early_stop_callback_VAE = EarlyStopping(
    monitor='val_loss',
    min_delta=hparams.VAE_min_delta,
    patience=10,
    verbose=False,
    mode='min'
)

checkpoint_callback_VAE = ModelCheckpoint(
    dirpath=os.path.dirname(
        Path(os.getcwd(), "models/encoder/VAE_loss/test/")),
    save_top_k=1,
    verbose=False,
    monitor='val_loss',
    mode='min',
    prefix='',
    filename=model_ID
)

#### Classifier Callbacks ####

early_stop_callback_cla = EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,
    patience=10,
    verbose=False,
    mode='min'
)
