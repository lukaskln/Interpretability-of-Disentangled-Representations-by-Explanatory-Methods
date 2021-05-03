#### Setup ####
from pytorch_lightning.loggers import WandbLogger
import warnings
import datetime
import os
import glob
from pathlib import Path

import pytorch_lightning as pl

warnings.filterwarnings('ignore')

print("Loading Modules...")
from src.__init__ import *

def run():

    ## Parser and Seeding ##

    parser = get_parser()
    hparams = parser.parse_args()

    print("Model ID:", hparams.model_ID)

    pl.seed_everything(hparams.seed)

    path_ckpt = Path(__file__).resolve().parents[0] / "models/encoder/VAE_loss/" / ("VAE_" + str(hparams.model_ID) + ".ckpt")

    if os.path.exists(path_ckpt) == True:
        print("[ERROR] Model ID does already exist")
        raise SystemExit(0)

    #### Logging ####

    if hparams.logger == "wandb":
        import wandb
        #wandb.init(project="VAE-mnist")
        logger = WandbLogger(project='VAE-mnist')
    else:
        logger = False

    #### Dataset selection ####
    print("Loading Datasets...")

    if hparams.dataset=="mnist":
        datamodule = datamodule_mnist
        input_height = 784
        num_classes = 10
    elif hparams.dataset == "dSprites":
        datamodule = datamodule_dSprites
        input_height = 4096
        num_classes = 3
    elif hparams.dataset == "OCT":
        datamodule = datamodule_OCT
        input_height = 40804
        num_classes = 4
    #### Select Encoder ####

    if hparams.model=="betaVAE_MLP":
            model_enc = betaVAE(
            beta=hparams.VAE_beta,
            lr=hparams.learning_rate,
            latent_dim=hparams.latent_dim,
            input_height=input_height
        )
    elif hparams.model == "betaVAE_VGG":
        model_enc = betaVAE_VGG(
            beta=hparams.VAE_beta,
            lr=hparams.learning_rate,
            latent_dim=hparams.latent_dim,
            input_height=input_height,
            dataset=hparams.dataset
        )
    elif hparams.model == "betaVAE_ResNet":
        model_enc = betaVAE_ResNet(
            beta=hparams.VAE_beta,
            lr=hparams.learning_rate,
            latent_dim=hparams.latent_dim,
            input_height=input_height,
            dataset=hparams.dataset
        )
    elif hparams.model == "betaTCVAE_MLP":
        model_enc = betaTCVAE(
            beta=hparams.VAE_beta,
            alpha=hparams.VAE_alpha,
            gamma=hparams.VAE_gamma,
            lr=hparams.learning_rate,
            latent_dim=hparams.latent_dim,
            input_height=input_height,
            dataset=hparams.dataset
        )
    elif hparams.model == "betaTCVAE_VGG":
        model_enc = betaTCVAE_VGG(
            beta=hparams.VAE_beta,
            alpha=hparams.VAE_alpha,
            gamma=hparams.VAE_gamma,
            lr=hparams.learning_rate,
            latent_dim=hparams.latent_dim,
            input_height=input_height,
            dataset=hparams.dataset
        )
    elif hparams.model == "betaTCVAE_ResNet":
        model_enc = betaTCVAE_ResNet(
            beta=hparams.VAE_beta,
            alpha=hparams.VAE_alpha,
            gamma=hparams.VAE_gamma,
            lr=hparams.learning_rate,
            latent_dim=hparams.latent_dim,
            input_height=input_height,
            dataset=hparams.dataset,
            pretrained = hparams.pretrained
        )
    else:
        raise Exception('Unknown Encoder type: ' + hparams.model)

    ## Training Encoder ##

    print("GPUs used:", torch.cuda.device_count())

    trainer = pl.Trainer(
        max_epochs=hparams.max_epochs,
        progress_bar_refresh_rate=25,
        callbacks=False,
        gpus=-1 if torch.cuda.device_count() > 1 else 0,
        distributed_backend="ddp" if torch.cuda.device_count() > 1 else False,
        sync_batchnorm=True if torch.cuda.device_count() > 1 else False,
        logger=logger,
        accelerator='ddp2'
    )


    trainer.fit(model_enc,
        datamodule.train_dataloader(), 
        datamodule.val_dataloader(),
        )
    
    trainer.save_checkpoint(path_ckpt)

    #### Select Classifier ####    

    if hparams.model_cla == "MLP":
        model_reg = MLP(input_dim=hparams.latent_dim,
                        num_classes=num_classes,
                        VAE_type= hparams.model,
                        path_ckpt=path_ckpt)
    elif hparams.model_cla == "reg":
        model_reg = LogisticRegression(
            input_dim=hparams.latent_dim,
            num_classes=num_classes,
            VAE_type=hparams.model,
            path_ckpt=path_ckpt)
    else:
        raise Exception('Unknown Classifer type: ' + hparams.cla_type)


    ## Training Classifier ##
    trainer = pl.Trainer(
        logger= logger,
        progress_bar_refresh_rate=25,
        callbacks=[early_stop_callback_cla, checkpoint_callback_cla],
        gpus=-1 if torch.cuda.device_count() > 1 else 0,
        distributed_backend="ddp" if torch.cuda.device_count() > 1 else False,
        sync_batchnorm=True if torch.cuda.device_count() > 1 else False
    )

    trainer.fit(model_reg, 
        datamodule.train_dataloader_cla(), 
        datamodule.val_dataloader_cla()
        )
                
    trainer.test(model_reg, datamodule.test_dataloader())


    if hparams.logger == "wandb":
        wandb.finish()


    ## Remove interim models ##

    path_ckpt_VAE = Path(os.getcwd(), "models/encoder/VAE_loss/",("VAE_" + str(hparams.model_ID) + ".ckpt"))
    path_ckpt_cla = Path(os.getcwd(), "models/classifier/",("cla_" + str(hparams.model_ID) + ".ckpt"))

    if hparams.save_model == False:
        os.remove(str(path_ckpt_VAE))
        os.remove(str(path_ckpt_cla))


if __name__ == '__main__':
    run()
