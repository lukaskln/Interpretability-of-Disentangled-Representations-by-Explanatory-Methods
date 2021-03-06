<br />
<p align="center">
  <a href="https://github.com/lukaskln/Interpretability-of-Disentangled-Representations-by-Explanatory-Methods">
    <img src="https://github.com/lukaskln/Interpretability-of-Disentangled-Representations-by-Explanatory-Methods/blob/master/images/OCT_LSF.gif" alt="Logo" width="170"> 
  </a>

  <h3 align="center">Interpretability of Disentangled Representations by Explanatory Methods</h3>

  <p align="center">
    Master Thesis at the ETH Zürich & University of Geneva
    <br />
    <a href="https://github.com/lukaskln/Interpretability-of-Disentangled-Representations-by-Explanatory-Methods/tree/master/src"><strong>Explore the Project »</strong></a>
    <br />
    <br />
    <a href="https://github.com/lukaskln/Interpretability-of-Disentangled-Representations-by-Explanatory-Methods/issues">Report Bug</a>
  </p>
</p>

## Table of Contents
* [About the Thesis](#about-the-thesis)
* [Folder Structure](#folder-structure)
* [Datasets](#datasets)
* [Usage](#usage)
  * [Run the code](#run-the-code)
  * [Reproducing the results](#reproducing-the-results)
* [Contact](#contact)

## About the Thesis

<p align="center">
  <img src="https://github.com/lukaskln/Interpretability-of-Disentangled-Representations-by-Explanatory-Methods/blob/master/images/Example_2.png" width="450"> 
</p>

AI engineers try to establish trust between humans and machines by either building interpretable models or applying explanatory methods upon black-box models. They form the two different bases to approach explainable AI. This thesis combines interpretable models and explanatory methods to gain new insights into their synergies in a semi-supervised setting, focusing on medical application. In medicine, AI models show astonishing prediction accuracies. However, they lack two crucial attributes for daily use in cooperation with a physician: First, the ability to be trained on small-sized labeled datasets, and second, to express the reasoning behind a prediction to challenge its diagnosis. The thesis demonstrates that both obstacles can be solved jointly through the semi-supervised setting, based on two proof-of-concept and one medical dataset containing OCT scans of the retina:

1. An encoder is trained unsupervised on an augmented VAE-loss to capture the independent data generating factors in the images, producing disentangled, human interpretable, latent space representations.

2. Based on these representations, a downstream classification task is solved by a neural network.

3. The attribution is measured by an explanatory method from the input image and the disentangled representation into the predictions. Further, the attribution from the input image into the disentangled representation is measured to explain the encoder.

## Folder Structure
```
├── README.md                                
├── LICENSE.txt                             
├── environment.yml                       - YAML file with the conda env.
├── main_evaluation.py                    - Main script to execute for evaluation
├── main.py                               - Main script to execute for training
├─── data                                 - Data storage folders (each filled after first run)
│    ├─── dSprites
│    ├─── mnist
│    └─── OCT
├─── images                               - Image export folder 
│
├─── models                               - Trained and saved models
│    ├─── classifier                      - Classifier checkpoints
│    └─── encoder
│         └─── VAE_loss                   - Encoder checkpoints
├─── src
│    │ __init__.py                        - Imports for main*.py scripts
│    │
│    ├─── download
│    │      dataimport.py                 - Initializes all datamodules
│    │      dataloader_dSprites.py        - dSprites datamodule
│    │      dataloader_mnist.py           - MNIST datamodule
│    │      dataloader_OCT.py             - OCT retina datamodule
│    │
│    ├─── evaluation
│    │      scores_attribution_methods.py - Attribution scores computation
│    │      vis_attribution_methods.py    - Attribution visualization
│    │      vis_latentspace.py            - Latent space visualization
│    │      vis_reconstructions.py        - Original & reconstruction images
│    │
│    └─── models
│         ├─── classifier                 - Classifier modules
│         │      cla_CNN.py
│         │      cla_logReg.py
│         │      cla_MLP.py
│         └─── encoder
│              └─── VAE_loss              - VAE-loss based encoder/decoder modules
│                     betaTCVAE.py
│                     betaTCVAE_ResNet.py
│                     betaTCVAE_VGG.py
│                     betaVAE.py
│                     betaVAE_ResNet.py
│                     betaVAE_VGG.py
│
└─── tools
       argparser.py                       - Parser for command line arguments 
       beta_effect.R                      - Beta effect plot
       callbacks.py                       - Early stopping and checkpoints
       confusion_matrix.R                 - Confusion matrix plot
       disentanglement_scores.R           - Importance plot and multinominal regression
```
## Datasets

To develop architectures, showcase theories, and test them on real-world medical data, three datasets can be selected in this library: two POC datasets and one medical. 

MNIST by [LeCun et al.](http://yann.lecun.com/exdb/mnist/), a famous database of handwritten digits, and dSprites, an artificially generated dataset of sprites to measure disentanglement by [Matthey et al.](https://github.com/deepmind/dsprites-dataset/) at DeepMind, are the two POC datasets. To test if the concepts also transfer to the medical domain and real world applications, the OCT retina dataset (V3) is selected, collected by [Kermany et al.](https://www.sciencedirect.com/science/article/pii/S0092867418301545) and consisting of 108,312 2D OCT images of the eye’s retina. The three different diseases and the normal state characterizing the retina OCT scans, are observed below:

<p align="center">
  <img src="https://github.com/lukaskln/Interpretability-of-Disentangled-Representations-by-Explanatory-Methods/blob/master/images/Example_1.jpg" width="600"> 
</p>

## Usage

All essential libraries for the execution of the code are provided in the environment.yml file from which a new conda environment can be created (Linux only). For the R scripts, please install the corresponding libraries beforehand. 

### Run the code

Once the virtual environment is activated, the code can be run as follows:

- Go into the main directory.
  ```sh
  cd Interpretability-of-Disentangled-Representations-by-Explanatory-Methods/
  ```
- Run the library with any number of custom arguments as defined in `tools/argparser.py`. 

  For example, when training a model:
  ```sh
  python main.py --model_ID=1001 --dataset="mnist" --model="betaTCVAE_VGG" --VAE_beta=4 --save_model=True
  ```
  And when evaluating the same model:
    ```sh
    python main_evaluation.py --model_ID=1001 --dataset="mnist" --model="betaTCVAE_VGG" --eval_latent_range=18 --num_workers=2
    ```
  Note that to save a model with a specific ID, the `save_model` argument must be set equal to True. Otherwise, the model will be deleted after the testing and can not be evaluated.

- If you want to run the code on ETH's Leonhard cluster, submit the same training job as above as follows:
  ```sh
  bsub -W 24:00 -n 8 -R "rusage[ngpus_excl_p=2,mem=4500]" "python main.py --model_ID=1001 --dataset='mnist' --model='betaTCVAE_VGG' --VAE_beta=4 --save_model=True"
  ```
  Load the proxy module on the Leonhard cluster via `module load eth_proxy` to download the data from external sources. The training code automatically supports data-parallel GPU training, with up to eight GPUs on one Leonhard node. The evaluation is computed exclusively on the CPU, supporting as many processors as possible. 

### Reproducing the results

To reproduce the results for downstream task performance (on different seeds) and the graphics in section five of the thesis, run the following code:

#### **Training**

For the &#946;-TCVAE loss based model with &#946; = 4 and a VGG encoder architecture:

- MNIST:
  ```sh
  python main.py --model_ID=1001 --dataset="mnist" --model="betaTCVAE_VGG" --VAE_beta=4 --batch_size=200 --save_model=True
  ```
- dSprites:
  ```sh
  python main.py --model_ID=2001 --dataset="dSprites" --model="betaTCVAE_VGG" --VAE_beta=4 --batch_size=200 --save_model=True
  ```
- OCT retina:
  ```sh
  python main.py --model_ID=3001 --dataset="OCT" --model="betaTCVAE_VGG" --VAE_beta=4 --learning_rate=0.0001 --batch_size=140 --latent_dim=20 --save_model=True
  ```

#### **Evaluation**

<p align="center">
  <img src="https://github.com/lukaskln/Interpretability-of-Disentangled-Representations-by-Explanatory-Methods/blob/master/images/Example_3.jpg" width="390">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://github.com/lukaskln/Interpretability-of-Disentangled-Representations-by-Explanatory-Methods/blob/master/images/Example.jpg" width="270"> 
</p>

The models used for the visualizations in the thesis in sections 5.2 and 5.3 are already provided as pre-trained models with saved weights and can be downloaded [here](https://polybox.ethz.ch/index.php/s/5YN0ESOCWB81Gcg/download). Copy the respective .ckpt files into the `models/classifier` (cla_\*.ckpt) and `models/encoder/VAE_loss` (VAE_\*.ckpt) folder. The models can be evaluated via the following IDs: 1000, 2000, and 3000.

- MNIST:
  ```sh
  python main_evaluation.py --model_ID=1000 --dataset="mnist" --method="IG" --model="betaTCVAE_VGG" --eval_latent_range=15 --num_workers=20
  ```
- dSprites:
  ```sh
  python main_evaluation.py --model_ID=2000 --dataset="dSprites" --method="EG" --model="betaTCVAE_VGG" --eval_latent_range=15 --num_workers=30 --batch_size=200
  ```
- OCT retina:
  ```sh
  python main_evaluation.py --model_ID=3000 --dataset="OCT" --method="EG" --model="betaTCVAE_VGG" --eval_latent_range=15 --num_workers=40 --batch_size=200
  ```
The `method` argument lets you choose betwen Integrated Gradients, Expected Gradients, Deep SHAP and Kernel SHAP as the explanatory method to compute the attributions. Set the `num_workers` argument to a suitable number for your computer, or remove it to choose half of your logical processors by default. Modify the `eval_latent_range` argument to change the weighting of the standard deviation when sampling through the latent conditional densities to get more samples from the tails or around the center of the normal distributions. 

#### **Supervised-Training**

A CNN with VGG16_bn architecture can be trained in a supervised setting on the small labeled dataset, for comparing the results to the semi-supervised setting:

  ```sh
  python main.py --model_ID=3002 --dataset="OCT" --model_cla="CNN" 
  ```

This is also possible for all other classifiers, as presented in section five of the thesis.

#### **Transfer-Learning and Pre-Training**

The classifier can be fine-tune on the supervised downstream task without training an unsupervised VAE-loss based model beforehand by using a transfer-learned encoder. The encoder is pre-trained on ImageNet and fixed in weights:

  ```sh
  python main.py --model_ID=3003 --dataset="OCT" --TL=True --model_TL="Inception" 
  ```
It is also possible to pre-train an encoder on the original data and then fine-tune it with the initialized weights to the downstream task:

  ```sh
  python main.py --model_ID=3004 --dataset="OCT" --model="betaVAE_VGG" --fix_weights=False  --latent_dim=20
  ```
  Pre-trained, and transfer-learned models can also be evaluated. The code above can be used to reproduce the results in section 6.1 of the thesis.
## Contact

Lukas Klein - [LinkedIn](https://www.linkedin.com/in/lukasklein1/) - lukas.klein@etu.unige.ch

