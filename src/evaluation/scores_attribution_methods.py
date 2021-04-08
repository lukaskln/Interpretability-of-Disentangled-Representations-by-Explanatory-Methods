import torch
import numpy as np
import shap
import matplotlib.pyplot as plt
from itertools import tee

class scores_AM_Original:
    def __init__(self, model, datamodule, type, out_dim=10, n=5):
        self.model = model
        self.model.eval()
        self.n = n
        self.out_dim = out_dim

        self.datamodule = datamodule
        self.type = type

    def deep_shap(self):
        print('\n Attribution of Original Images:')

        iter_obj = iter(self.datamodule)
        a, _ = iter_obj.next()
        images_test, labels_test = iter_obj.next()
        b, _ = iter_obj.next()
        c, _ = iter_obj.next()
        d, _ = iter_obj.next() # messy solution, change that!
        images_train = torch.cat([a,b,c,d], dim = 0)

        height = images_test[0].shape[1]
        width = images_test[0].shape[2]

        if self.type == "betaVAE" or self.type == "betaTCVAE":
            images_test = images_test[:self.n].view(-1, height * width)
            background = images_train.view(-1, height * width)
        else:
            images_test = images_test[:self.n]
            background = images_train

        model = self.model
        # estimate expectation with one batch size
        exp = shap.DeepExplainer(model, data = background)
        deep_shap_values = exp.shap_values(images_test, check_additivity=True)

        if self.type == 'betaVAE' or self.type == 'betaTCVAE':
            deep_shap_values = np.asarray(deep_shap_values).reshape(
                self.out_dim, self.n, height, width)
            images_test = images_test.view(-1, 1, height, width)

        return deep_shap_values, images_test

    def expgrad_shap(self):
        print('\n Attribution of Original Images:')

        with torch.no_grad():
            iter_obj = iter(self.datamodule)
            iter_obj.next()
            images_test, labels_test = iter_obj.next()

            height = images_test[0].shape[1]
            width = images_test[0].shape[2]

            if self.type == "betaVAE" or self.type == "betaTCVAE":
                images_test = images_test[:self.n].view(-1, height * width)
                background = torch.zeros((1, height * width))
            else:
                images_test = images_test[:self.n]
                background = torch.zeros((1,1,height,width))

        exp = shap.GradientExplainer(self.model,
                                    data=background
                                    )

        expgrad_shap_values = exp.shap_values(images_test)

        if self.type == 'betaVAE' or self.type == 'betaTCVAE':
            expgrad_shap_values = np.asarray(expgrad_shap_values).reshape(
                self.out_dim, self.n, height, width)
            images_test = images_test.view(-1, 1, height, width)

        return expgrad_shap_values, images_test


class scores_AM_Latent:
    def __init__(self, model, encoder, datamodule, type):
        self.model = model
        self.model.eval()

        self.encoder = encoder
        self.encoder.eval()

        self.datamodule = datamodule
        self.type = type

    def expgrad_shap(self):
        print("\n Attribution of Latent Space Representations: ")

        with torch.no_grad():
            iter_obj = iter(self.datamodule)
            images_train, labels_train = iter_obj.next()
            images_test, labels_test = iter_obj.next()

            height = images_train[0].shape[1]
            width = images_train[0].shape[2]

            if self.type == "betaVAE" or self.type == "betaTCVAE":
                images_train = images_train.view(-1, height * width)
                images_test = images_test.view(-1, height * width)

            encoder = self.encoder

            encoding_train, _ = encoder.encode(images_train)
            encoding_test, _ = encoder.encode(images_test)

        exp = shap.GradientExplainer(self.model, 
                                    data = encoding_train)
        
        expgrad_shap_values = exp.shap_values(encoding_test)
        return exp, expgrad_shap_values, encoding_test.numpy().astype('float32'), labels_test

    def deep_shap(self):
        print("\n Attribution of Latent Space Representations: ")

        with torch.no_grad():
            iter_obj = iter(self.datamodule)
            a, _ = iter_obj.next()
            images_test, labels_test = iter_obj.next()
            b, _ = iter_obj.next()
            c, _ = iter_obj.next()
            d, _ = iter_obj.next()  # messy sol
            images_train = torch.cat([a, b, c, d], dim=0)

            height = images_test[0].shape[1]
            width = images_test[0].shape[2]

            if self.type == "betaVAE" or self.type == "betaTCVAE":
                images_train = images_train.view(-1, height * width)
                images_test = images_test.view(-1, height * width)
        
            encoder = self.encoder

            encoding_train, _ = encoder.encode(images_train)
            encoding_test, _ = encoder.encode(images_test)

        exp = shap.DeepExplainer(self.model, encoding_train)
        deep_shap_values = exp.shap_values(encoding_test, check_additivity=True)

        return exp, deep_shap_values, encoding_test.numpy().astype('float32'), labels_test

