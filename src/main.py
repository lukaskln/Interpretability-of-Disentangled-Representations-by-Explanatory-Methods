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

print("Loading Modules...")

from src.__init__ import *

print("Model ID:", model_ID)

## Parser and Seeding ##

parser = get_parser()
hparams = parser.parse_args()

pl.seed_everything(hparams.seed)

#### Logging ####

if hparams.logger == True:
    import wandb
    #wandb.init(project="VAE-mnist")
    wandb_logger = WandbLogger(project='VAE-mnist')
else:
    wandb_logger = False

#### Dataset selection ####
print("Loading Datasets...")

if hparams.dataset=="mnist":
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

#### Select Encoder ####

if hparams.VAE_type=="betaVAE_MLP":
        model_enc = betaVAE(
        beta=hparams.VAE_beta,
        lr=hparams.VAE_lr,
        latent_dim=hparams.VAE_latent_dim,
        input_height=input_height
    )
elif hparams.VAE_type == "betaVAE_CNN":
    model_enc = betaVAE_CNN(
        beta=hparams.VAE_beta,
        lr=hparams.VAE_lr,
        latent_dim=hparams.VAE_latent_dim,
        c=hparams.CNN_capacity,
        input_height=input_height,
        dataset=hparams.dataset
    )
elif hparams.VAE_type == "betaVAE_ResNet":
    model_enc = betaVAE_ResNet(
        beta=hparams.VAE_beta,
        lr=hparams.VAE_lr,
        latent_dim=hparams.VAE_latent_dim,
        c=hparams.CNN_capacity,
        input_height=input_height,
        dataset=hparams.dataset
    )
elif hparams.VAE_type == "betaTCVAE_MLP":
    model_enc = betaTCVAE(
        beta=hparams.VAE_beta,
        alpha=hparams.TCVAE_alpha,
        gamma=hparams.TCVAE_gamma,
        lr=hparams.VAE_lr,
        latent_dim=hparams.VAE_latent_dim,
        input_height=input_height
    )
elif hparams.VAE_type == "betaTCVAE_CNN":
    model_enc = betaTCVAE_CNN(
        beta=hparams.VAE_beta,
        alpha=hparams.TCVAE_alpha,
        gamma=hparams.TCVAE_gamma,
        lr=hparams.VAE_lr,
        latent_dim=hparams.VAE_latent_dim,
        c=hparams.CNN_capacity,
        input_height=input_height,
        dataset=hparams.dataset
    )
elif hparams.VAE_type == "betaTCVAE_ResNet":
    model_enc = betaTCVAE_ResNet(
        beta=hparams.VAE_beta,
        alpha=hparams.TCVAE_alpha,
        gamma=hparams.TCVAE_gamma,
        lr=hparams.VAE_lr,
        latent_dim=hparams.VAE_latent_dim,
        c=hparams.CNN_capacity,
        input_height=input_height,
        dataset=hparams.dataset
    )
else:
    raise Exception('Unknown Encoder type: ' + hparams.VAE_type)

## Training Encoder ##

trainer = pl.Trainer(
    max_epochs=hparams.VAE_max_epochs,
    progress_bar_refresh_rate=25,
    callbacks=[checkpoint_callback_VAE],
    gpus=torch.cuda.device_count()
)


trainer.fit(model_enc, datamodule_enc)


#### Select Classifier ####    

if hparams.cla_type == "MLP":
    model_reg = MLP(input_dim=hparams.VAE_latent_dim,
                    num_classes=num_classes,
                    VAE_type= hparams.VAE_type,
                    learning_rate=hparams.cla_lr)
elif hparams.cla_type == "reg":
    model_reg = LogisticRegression(
        input_dim=hparams.VAE_latent_dim,
        num_classes=num_classes,
        VAE_type=hparams.VAE_type,
        learning_rate=hparams.cla_lr)
else:
    raise Exception('Unknown Classifer type: ' + hparams.cla_type)


## Training Classifier ##
trainer = pl.Trainer(
    logger=wandb_logger,
    progress_bar_refresh_rate=25,
    callbacks=[early_stop_callback_cla, checkpoint_callback_cla],
    gpus=torch.cuda.device_count()
)

trainer.fit(model_reg, dataloader_train, dataloader_val)
            
trainer.test(model_reg, dataloader_test)


if hparams.logger == True:
    wandb.finish()


## Remove interim models ##

path_ckpt_VAE = Path(os.getcwd(), "models/encoder/VAE_loss/",("VAE_" + model_ID + ".ckpt"))
path_ckpt_cla = Path(os.getcwd(), "models/classifier/",("cla_" + model_ID + ".ckpt"))

if hparams.save_model == False:
    os.remove(str(path_ckpt_VAE))
    os.remove(str(path_ckpt_cla))

