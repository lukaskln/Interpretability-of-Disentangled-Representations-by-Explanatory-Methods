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
    parser = ArgumentParser(description='semi-supervised learning')
    # Data and Computation Parameters
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--dataset", choices=["mnist","dSprites","OCT"], default="mnist", type=str)
    
    # Training Parameters
    parser.add_argument("--model", choices=["betaVAE_MLP", "betaVAE_VGG", "betaVAE_ResNet",
                                            "betaTCVAE_MLP", "betaTCVAE_VGG", "betaTCVAE_ResNet"], 
                                            default="betaVAE_MLP", type=str)
    parser.add_argument("--model_cla", choices=["MLP", "reg"], default="MLP", type=str)
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument("--max_epochs", default=70, type=int)
    parser.add_argument("--min_delta", default=0.001, type=float)
    parser.add_argument("--latent_dim", default=10, type=int)
    parser.add_argument("--VAE_beta", default=1, type=int)
    parser.add_argument("--VAE_gamma", default=1, type=int)
    parser.add_argument("--VAE_alpha", default=1, type=int)
    
    # Logging and Reproducibility 
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--logger", choices=["_", "wandb", "tensorboard"], default="_", type=str)
    parser.add_argument("--save_model", default=False, type=str2bool)
    parser.add_argument("--pretrained", default=False, type=str2bool)
    
    # Evaluation
    parser.add_argument("--eval_model_ID", default=1000, type=int) 
    parser.add_argument("--eval_latent_range", default=3.0, type=float)
    return parser
