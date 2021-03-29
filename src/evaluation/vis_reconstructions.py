import torchvision.utils
import numpy as np
import matplotlib.pyplot as plt
import torch
plt.ion()

class vis_Reconstructions:
    def __init__(self, model, datamodule, type):
        self.model = model
        self.model.eval()

        self.datamodule = datamodule
        self.type = type

    def to_img(self, x):
        x = x.clamp(0, 1)
        return x

    def show_image(self, img):
        img = self.to_img(img)
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    def visualise_output(self, images):

        with torch.no_grad():
            
            height = images[0].shape[1]
            width = images[0].shape[2]

            model = self.model

            if self.type=='betaVAE' or self.type=='betaTCVAE':
                mu, log_var = model.encode(images.view(-1, height * width))
            else:
                mu, log_var = model.encode(images)


            z = model.sampling(mu, log_var)
            images = model.decode(z)
            images = images.cpu()

            if self.type == 'betaVAE' or self.type == 'betaTCVAE':
                images = self.to_img(images.view(-1, 1, height, width))
            else:
                images = self.to_img(images)
            
            np_imagegrid = torchvision.utils.make_grid(images[0:50], 10, 5).numpy()
            plt.imshow(np.transpose(np_imagegrid, (1, 2, 0)))
            plt.axis('off')
        return plt.show()

    def visualise(self):

        images, labels = iter(self.datamodule).next()

        print('Original images:')
        self.show_image(torchvision.utils.make_grid(images[0:50], 10, 5))
        plt.axis('off')
        plt.show()

        print('VAE reconstructions:')
        self.visualise_output(images)

