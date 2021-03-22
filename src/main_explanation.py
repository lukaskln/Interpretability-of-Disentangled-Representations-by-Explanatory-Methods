#### Setup ####
from pytorch_lightning.loggers import WandbLogger
import warnings
import datetime
import os
import glob
from pathlib import Path

main_dir = os.path.dirname(Path(__file__).resolve().parents[0])
os.chdir(main_dir)

import pytorch_lightning as pl

# warnings.filterwarnings('ignore')

from src.__init__ import *