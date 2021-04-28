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

        shap.image_plot(shap_values, test_values)

class vis_AM_Latent:
    def __init__(self, shap_values, explainer, encoding_test, labels_test):
        self.shap_values = shap_values
        self.encoding_test = encoding_test
        self.labels_test = labels_test
        self.exp = explainer
        self.n = len(np.unique(labels_test.numpy()))

        if self.n == 10:
            self.labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        elif self.n == 4:
            self.labels = ["CNV", "DME", "Drusen","Normal"]
        else:
            self.labels = ["square", "ellipse", "heart"]

    def visualise(self):
        print("\n Total Latent Feature Attribution:")

        shap.summary_plot(self.shap_values, 
                          self.encoding_test, 
                          plot_type="bar",
                          color= plt.cm.tab10,
                          class_names=self.labels
                        )

        print("\n Attribution of Latent Features:")

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
        axes = axes.ravel()

        for i in range(0, 4, 1):
            plt.subplot(2, 2, i+1)
            shap.multioutput_decision_plot(
                np.zeros((1,self.n)).tolist()[0], #[i for i in softmax(self.exp.expected_value)]
                self.shap_values,
                highlight=self.labels_test[i],
                legend_labels=self.labels,
                legend_location='lower right',
                show=False,
                auto_size_plot=False,
                row_index=i)

        plt.show()


class vis_AM_Latent_on_Rec:
    def __init__(self, shap_values, encoding_test, model, type):
        self.shap_values = torch.from_numpy(np.stack(shap_values)[:, 0:5, :])
        self.encoding_test = torch.from_numpy(encoding_test[0:5,:]).type(torch.DoubleTensor)
        self.model = model
        self.type = type

    def to_img(self, x):
        x = x.clamp(0, 1)
        return x

    def rec_image(self):
        with torch.no_grad():
            images = self.model.decode(self.encoding_test.float())
            images = images.cpu()

        if self.type == 'betaVAE' or self.type == 'betaTCVAE':
            images = self.to_img(images.view(-1, 1, 28, 28))
        else:
            images = self.to_img(images)

        return images

    def rec_shap(self):
        shap_list = []
        for i in range(0,self.shap_values.shape[1]):
            with torch.no_grad():
                value = self.model.decode(self.shap_values[:, i, :].float())
                value = value.cpu()

            if self.type == 'betaVAE' or self.type == 'betaTCVAE':
                shap_list.append(value.view(-1, 28, 28))
            else:
                shap_list.append(value)

        shap_full = torch.stack((shap_list[0:self.shap_values.shape[1]]), dim=1)

        return shap_full

    def prep(self, shap_values, test_images):
        shap_numpy = [np.swapaxes(np.swapaxes(s.numpy(), 1, -1), 1, 2) for s in shap_values]
        test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)
        return shap_numpy, test_numpy

    def visualise(self):
        print("\n Attribution of Latent Features on Reconstructions:")

        test_images_rec = self.rec_image()
        shap_values_rec = self.rec_shap()

        shap_values, test_values = self.prep(shap_values_rec, test_images_rec)

        shap.image_plot(shap_values, test_values)
