import torchvision.utils
import numpy as np
import matplotlib.pyplot as plt
import torch
import shap
from scipy.special import softmax, logit, expit
import colorcet as cc

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

        shap.image_plot(shap_values, -test_values)

class vis_AM_Latent:
    def __init__(self, shap_values, explainer, encoding_test, labels_test):
        self.shap_values = shap_values
        self.encoding_test = encoding_test
        self.labels_test = labels_test
        self.exp = explainer

        #shap.initjs()

    def visualise(self):
        print("\n Total Latent Feature Attribution:")

        shap.summary_plot(self.shap_values, 
                          self.encoding_test, 
                          plot_type="bar",
                          color= plt.cm.tab10
                        )

        print("\n Attribution of Latent Features:")

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
        axes = axes.ravel()

        for i in range(0, 4, 1):
            plt.subplot(2, 2, i+1)
            shap.multioutput_decision_plot(
                np.zeros((1,10)).tolist()[0], #[i for i in softmax(self.exp.expected_value)]
                self.shap_values,
                highlight=self.labels_test[i],
                legend_labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                legend_location='lower right',
                show=False,
                auto_size_plot=False,
                row_index=i)

        plt.show()


