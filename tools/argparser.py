import argparse
from argparse import ArgumentParser

"""
This file contains the declaration of our argument parser
"""

# Needed to parse booleans from command line properly
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_parser():
    parser = ArgumentParser(description='master thesis')
    # General Model Parameters
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--logger", default=False, type=str2bool)
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--save_model", default=False, type=str2bool)
    parser.add_argument("--pretrained", default=False, type=str2bool)
    parser.add_argument("--dataset", choices=["mnist", "mnist_small","dSprites_small","OCT_small"], default="mnist", type=str)
    # VAE
    parser.add_argument("--VAE_type", choices=["betaVAE_MLP", "betaVAE_VGG", "betaVAE_ResNet",
                                               "betaTCVAE_MLP", "betaTCVAE_VGG", "betaTCVAE_ResNet"], 
                                               default="betaVAE_MLP", type=str)
    parser.add_argument("--VAE_beta", default=1, type=int)
    parser.add_argument("--VAE_lr", default=0.001, type=float)
    parser.add_argument("--VAE_max_epochs", default=70, type=int)
    parser.add_argument("--VAE_min_delta", default=0.001, type=float)
    parser.add_argument("--VAE_latent_dim", default=10, type=int)
    parser.add_argument("--CNN_capacity", default=64, type=int)
    # beta TCVAE
    parser.add_argument("--TCVAE_gamma", default=1, type=int)
    parser.add_argument("--TCVAE_alpha", default=1, type=int)
    # Classifier
    parser.add_argument("--cla_type", choices=["MLP", "reg"], default="MLP", type=str)
    parser.add_argument("--cla_lr", default=0.001, type=float)
    # Evaluation
    parser.add_argument("--eval_model_ID", default=1000, type=int) 
    parser.add_argument("--eval_latent_range", default=3.0, type=float)
    return parser
