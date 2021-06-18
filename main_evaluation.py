#### Setup ####
import warnings
import datetime
import os
import glob
from pathlib import Path
import pathlib
import platform

import pytorch_lightning as pl
import torch

warnings.filterwarnings('ignore')

print("Loading Modules...")

from src.__init__ import *

from src.evaluation.vis_latentspace import *
from src.evaluation.vis_reconstructions import *
from src.evaluation.scores_attribution_methods import *
from src.evaluation.vis_attribution_methods import *

parser = get_parser()
hparams = parser.parse_args()

#### Model import ####

## Encoder ##
path_ckpt_VAE = Path("./models/encoder/VAE_loss/VAE_" + str(hparams.model_ID) + ".ckpt")

if os.path.exists(path_ckpt_VAE)==False:
    print("[ERROR] Model does not exist. Please check the /models folder for existing models.")
    raise SystemExit(0)

architectures_VAE = [
    betaTCVAE,
    betaTCVAE_VGG,
    betaTCVAE_ResNet,
    betaVAE,
    betaVAE_VGG,
    betaVAE_ResNet
]

for architecture in architectures_VAE:
    try:
        encoder = architecture.load_from_checkpoint(path_ckpt_VAE)
        encoder_type = architecture.__name__
        break
    except RuntimeError:
        # repeat the loop on failure
        continue

## Classifier ##

path_ckpt_cla = Path("./models/classifier/cla_" + str(hparams.model_ID) + ".ckpt")

if os.path.exists(path_ckpt_cla) == False:
    print("[ERROR] Model does not exist. Please check the /models folder for existing models.")
    raise SystemExit(0)

architectures_cla = [
    MLP,
    LogisticRegression
]

system = platform.system()
if system == "Windows":
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

for architecture in architectures_cla:
    try:
        cla = architecture.load_from_checkpoint(path_ckpt_cla)
        cla_type = architecture.__name__
        break
    except RuntimeError:
        # repeat the loop on failure
        continue

## Dataset selection ##
print("Loading Datasets...")

if hparams.dataset == "mnist":
    datamodule = datamodule_mnist
    input_height = 784
    num_classes = 10
elif hparams.dataset == "dSprites":
    datamodule = datamodule_dSprites
    input_height = 4096
    num_classes = 3
elif hparams.dataset == "OCT":
    datamodule = datamodule_OCT
    input_height = int(326 ** 2)
    num_classes = 4

#### Visualizations ####

try:
    mu, sd = vis_Reconstructions(encoder, datamodule.train_dataloader(), type=encoder_type).visualise()
except RuntimeError:
    print("[ERROR] Wrong dataset selected? Check --dataset=...")
    raise SystemExit(0)


vis_LatentSpace(encoder,
                latent_dim=encoder.state_dict()['fc_mu.weight'].shape[0],
                latent_range=hparams.eval_latent_range,
                input_dim=np.sqrt(input_height).astype(int),
                mu = mu,
                sd = sd
                ).visualise()

#### Attribution Methods ####

print('Visualizing Attribution of Original Images into Predictions...')
scores, test_images = scores_AM_Original(cla, 
                                        datamodule.train_dataloader(),
                                        type=encoder_type,
                                        out_dim = cla.state_dict()['fc2.weight'].shape[0]
                                        ).expgrad_shap()


vis_AM_Original(scores, test_images).visualise()
plt.savefig('./images/attribution original.png')

exp, scores, encoding_test, labels_test = scores_AM_Latent(model = cla,
                                        encoder = encoder,
                                        datamodule=datamodule.train_dataloader(),
                                        type = encoder_type,
                                        ).expgrad_shap()


vis_AM_Latent(shap_values=scores,
            explainer=exp, 
            encoding_test=encoding_test,
            labels_test=labels_test
            ).visualise()

print('Visualizing Attribution of Original Images into Latent Space Representations...')
scores, test_images = scores_AM_Original(encoder,
                                        datamodule.train_dataloader(),
                                        type=encoder_type,
                                        out_dim = encoder.state_dict()['fc_mu.weight'].shape[0]
                                        ).expgrad_shap()

vis_AM_Original(scores, test_images).visualise()
plt.savefig('./images/attribution original into LSF.png')
