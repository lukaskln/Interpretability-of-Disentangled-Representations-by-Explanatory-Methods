from torchvision.utils import make_grid
import torch
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

class vis_LatentSpace:
    def __init__(self, model, latent_dim=10, latent_range=3):
        self.model = model
        self.model.eval()

        self.latent_dim = latent_dim
        self.latent_range = latent_range

    def to_img(self, x):
        x = x.clamp(0, 1)
        return x

    def show_image(self, img):
        img = self.to_img(img)
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    def visualise(self):
        recon = []

        for i in range(0,self.latent_dim,1):
            latent = torch.zeros(self.latent_dim, 20)
            latent[i, :] = torch.linspace(-self.latent_range, self.latent_range, 20)
            latent = torch.transpose(latent, 0, 1)
            img_recon = self.model.decoder(latent)
            recon.append(img_recon.view(-1, 1, 28, 28))

        recon = torch.cat(recon)

        fig, ax = plt.subplots(figsize=(10, 8))
        plt.axis('off')
        self.show_image(make_grid(recon.data, 20, 8))

        for i in range(0, self.latent_dim, 1):
            plt.text(3, 15 + (i*36), str(i), color="red")

        print("Latent Space Features:")
        plt.show()

