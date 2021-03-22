#### Setup ####
import warnings
import datetime
import os
import glob
from pathlib import Path

main_dir = os.path.dirname(Path(__file__).resolve().parents[0])
os.chdir(main_dir)

path_ckpt = Path(__file__).resolve().parents[3] / "models/encoder/VAE_loss/Best_VAE.ckpt"


import pytorch_lightning as pl

# warnings.filterwarnings('ignore')

from src.__init__ import *

model_enc = betaVAE.load_from_checkpoint(path_ckpt)

#### Disentanglement scores ####





#### Attribution Methods ####

#### Visualizations ####

