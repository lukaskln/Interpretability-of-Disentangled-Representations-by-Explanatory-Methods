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
from src.evaluation.scores_attribution_methods import *
from src.evaluation.vis_attribution_methods import *

parser = get_parser()
hparams = parser.parse_args()

#### Model import ####

## Encoder ##
path_ckpt_VAE = Path(os.getcwd(), 
                "models/encoder/VAE_loss/", 
                ("VAE_" + str(hparams.eval_model_ID) + ".ckpt")
            )

if os.path.exists(path_ckpt_VAE)==False:
    print("[ERROR] Model does not exist. Please check the /models folder for existing models.")
    raise SystemExit(0)

architectures_VAE = [
    betaVAE,
    betaVAE_VGG,
    betaVAE_ResNet,
    betaTCVAE,
    betaTCVAE_VGG,
    betaTCVAE_ResNet
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

path_ckpt_cla = Path(os.getcwd(),
                     "models/classifier/",
                     ("cla_" + str(hparams.eval_model_ID) + ".ckpt")
                     )

if os.path.exists(path_ckpt_cla) == False:
    print("[ERROR] Model does not exist. Please check the /models folder for existing models.")
    raise SystemExit(0)

architectures_cla = [
    MLP,
    LogisticRegression
]

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
    num_classes = 3


#### Visualizations ####

try:
    vis_Reconstructions(encoder, dataloader_test, type=encoder_type).visualise()
except RuntimeError:
    print("[ERROR] Wrong dataset selected? Check --dataset=...")
    raise SystemExit(0)


vis_LatentSpace(encoder,
                latent_dim=encoder.state_dict()['fc_mu.weight'].shape[0],
                latent_range=hparams.eval_latent_range,
                input_dim=np.sqrt(input_height).astype(int),
                ).visualise()

#### Attribution Methods ####

print('\n Attribution of Original Images into Predictions:')
scores, test_images = scores_AM_Original(cla, 
                                         dataloader_test, 
                                         type=encoder_type,
                                         out_dim = cla.state_dict()['fc2.weight'].shape[0]
                                         ).expgrad_shap()


vis_AM_Original(scores, test_images).visualise()

exp, scores, encoding_test, labels_test = scores_AM_Latent(model = cla,
                                        encoder = encoder,
                                        datamodule = dataloader_test,
                                        type = encoder_type,
                                        ).expgrad_shap()


vis_AM_Latent(shap_values=scores,
              explainer=exp, 
              encoding_test=encoding_test,
              labels_test=labels_test
              ).visualise()

print('\n Attribution of Original Images into Latent Space Representations:')
scores, test_images = scores_AM_Original(encoder,
                                         dataloader_test, 
                                         type=encoder_type,
                                         out_dim = encoder.state_dict()['fc_mu.weight'].shape[0]
                                         ).expgrad_shap()

vis_AM_Original(scores, test_images).visualise()

# vis_AM_Latent_on_Rec(shap_values=scores, 
#                     encoding_test = encoding_test, 
#                     model = encoder,
#                     type = encoder_type
#                      ).visualise()



