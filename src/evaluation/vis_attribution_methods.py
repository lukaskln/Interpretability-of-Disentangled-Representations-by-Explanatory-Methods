import torchvision.utils
import numpy as np
import matplotlib.pyplot as plt
import torch
import shap
from scipy.special import softmax, logit, expit

class vis_AM_Original:
    def __init__(self, shap_values, test_images):
        self.shap_values = shap_values
        self.test_images = test_images

    def prep(self, shap_values, test_images):
        shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
        test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)
        return shap_numpy, test_numpy

    def visualise(self):

        shap_values, test_values = self.prep(self.shap_values, self.test_images)

        print('Attribution of Original Images:')
        shap.image_plot(shap_values, -test_values)

class vis_AM_Latent:
    def __init__(self, shap_values, explainer, encoding_test, labels_test, n = 0):
        self.shap_values = shap_values
        self.encoding_test = encoding_test
        self.labels_test = labels_test
        self.exp = explainer
        self.n = n

        shap.initjs()
        
    def force_plot(self):
        shap.force_plot(
            self.exp.expected_value[self.n],
            self.shap_values[0][self.n, :],
            link="logit")

    def visualise(self):
        pass



