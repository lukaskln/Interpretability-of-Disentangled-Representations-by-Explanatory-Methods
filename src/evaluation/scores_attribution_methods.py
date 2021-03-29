import torch
import numpy as np
import shap
import matplotlib.pyplot as plt

class scores_AM_Original:
    def __init__(self, model, datamodule, type, out_dim = 10, n=5):
        self.model = model
        self.model.eval()
        self.n = n
        self.out_dim = out_dim

        self.datamodule = datamodule
        self.type = type

    def baseline(self, images):
        height = images[0].shape[1]
        width = images[0].shape[2]
        baseline = torch.zeros(height, width)
        return baseline
    
    def deep_shap(self):
        iter_obj = iter(self.datamodule)
        images_1, _ = iter_obj.next()
        images_2, _ = iter_obj.next()

        height = images_1[0].shape[1]
        width = images_1[0].shape[2]

        if self.type == 'betaVAE' or self.type == 'betaTCVAE':
            background = images_1.view(-1, height * width)
            test_images = images_2[:self.n].view(-1, height * width)
        else:
            background = images_1
            test_images = images_2[:self.n]

        model = self.model
        # estimate expectation with one batch size
        exp = shap.DeepExplainer(model, background) 

        deep_shap_values = exp.shap_values(test_images)

        if self.type == 'betaVAE' or self.type == 'betaTCVAE':
            deep_shap_values = np.asarray(deep_shap_values).reshape(self.out_dim, self.n, height, width)
            test_images = test_images.view(-1, 1, height, width)

        return deep_shap_values, test_images




class scores_AM_Latent:
    def __init__(self, model, encoder, datamodule, type):
        self.model = model
        self.model.eval()

        self.encoder = encoder
        self.encoder.eval()

        self.datamodule = datamodule
        self.type = type

    def kernel_shap(self):
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

            exp = shap.KernelExplainer(self.model.forward_no_enc, 
                                       data = encoding_train.numpy().astype('float32'))
            
            kernel_shap_values = exp.shap_values(encoding_test.numpy().astype('float32'))
            return exp, kernel_shap_values, encoding_test.numpy().astype('float32'), labels_test


