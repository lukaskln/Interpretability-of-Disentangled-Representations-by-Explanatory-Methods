import os
from pathlib import Path
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

#### VAE Callbacks ####

early_stop_callback_VAE = EarlyStopping(
    monitor='vae_loss',
    min_delta=0.00001,
    patience=10,
    verbose=False,
    mode='min'
)

checkpoint_callback_VAE = ModelCheckpoint(
    dirpath=os.path.dirname(
        Path(os.getcwd(), "models/encoder/VAE_loss/test/")),
    save_top_k=1,
    verbose=False,
    monitor='vae_loss',
    mode='min',
    prefix='',
    filename="Best_VAE"
)

#### Classifier Callbacks ####

early_stop_callback_cla = EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,
    patience=10,
    verbose=False,
    mode='min'
)
