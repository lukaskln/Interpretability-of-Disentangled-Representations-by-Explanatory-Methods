#### Setup ####
from pytorch_lightning.loggers import WandbLogger
import warnings
import datetime
import os
import glob
from pathlib import Path

main_dir = os.path.dirname(Path(__file__).resolve().parents[0])
os.chdir(main_dir)

# TODO shell commands from script:
#import platform
#from subprocess import call
# plt = platform.system()

# if plt == "Linux": 
#     command = 'export PYTHONPATH="${PYTHONPATH}:/cluster/home/luklein/Semi-supervised-methods"'
#     process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
#     output, error = process.communicate()

import pytorch_lightning as pl

warnings.filterwarnings('ignore')

from src.__init__ import *

## Parser and Seeding ##

parser = get_parser()
hparams = parser.parse_args()

pl.seed_everything(hparams.seed)

## Remove all saved models ##

files = glob.glob(os.path.dirname(Path(os.getcwd(), "models/encoder/VAE_loss/*/test/"))
        )

for f in files:
    os.remove(f) # As administrator

#### Logging ####

if hparams.logger == True:
    import wandb
    #wandb.init(project="VAE-mnist")
    wandb_logger = WandbLogger(project='VAE-mnist')
else:
    wandb_logger = False

#### Select Encoder ####

if hparams.TCVAE==True:
    if hparams.VAE_CNN == False:
        model_enc = betaTCVAE(
            beta=hparams.VAE_beta,
            alpha=hparams.TCVAE_alpha,
            gamma=hparams.TCVAE_gamma,
            lr=hparams.VAE_lr,
            latent_dim=hparams.VAE_latent_dim
        )
    else: 
        model_enc = betaTCVAE_CNN(
            beta=hparams.VAE_beta,
            alpha=hparams.TCVAE_alpha,
            gamma=hparams.TCVAE_gamma,
            lr=hparams.VAE_lr,
            latent_dim=hparams.VAE_latent_dim,
            c=hparams.CNN_capacity
        )
else:
    if hparams.VAE_CNN == False:
        model_enc = betaVAE(
            beta=hparams.VAE_beta,
            lr=hparams.VAE_lr,
            latent_dim=hparams.VAE_latent_dim
        )
    else: 
        model_enc = betaVAE_CNN(
            beta=hparams.VAE_beta,
            lr=hparams.VAE_lr,
            latent_dim=hparams.VAE_latent_dim,
            c = hparams.CNN_capacity
        )

## Training Encoder ##

trainer = pl.Trainer(
    max_epochs=hparams.VAE_max_epochs,
    progress_bar_refresh_rate=25,
    callbacks=[checkpoint_callback_VAE],
    gpus=torch.cuda.device_count()
)

trainer.fit(model_enc, datamodule_mnist)

#### Select Classifier ####    

if hparams.cla_type == "MLP":
    model_reg = MLP(input_dim=hparams.VAE_latent_dim,
                    num_classes=10,
                    VAE_CNN=hparams.VAE_CNN,
                    TCVAE=hparams.TCVAE,
                    learning_rate=hparams.cla_lr)
elif hparams.cla_type == "reg":
    model_reg = LogisticRegression(
        input_dim=hparams.VAE_latent_dim,
        num_classes=10, 
        VAE_CNN=hparams.VAE_CNN,
        TCVAE=hparams.TCVAE,
        learning_rate=hparams.cla_lr)
else:
    raise Exception('Unknown Classifer type: ' + hparams.cla_type)


## Training Classifier ##
trainer = pl.Trainer(
    logger=wandb_logger,
    progress_bar_refresh_rate=25,
    callbacks=[early_stop_callback_cla], 
    gpus=torch.cuda.device_count()
)

trainer.fit(model_reg, datamodule_mnist.train_dataloader(), datamodule_mnist.val_dataloader())

trainer.test(model_reg, datamodule_mnist.test_dataloader())

if hparams.logger == True:
    wandb.finish()

