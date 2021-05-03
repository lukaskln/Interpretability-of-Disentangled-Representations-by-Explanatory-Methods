from torchvision.utils import make_grid
import torch
import numpy as np
import matplotlib.pyplot as plt
plt.ioff()

class vis_LatentSpace:
    def __init__(self, model, latent_dim=10, latent_range=3, input_dim=28):
        self.model = model
        self.model.eval()

        self.latent_dim = latent_dim
        self.latent_range = latent_range
        self.input_dim = input_dim

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
            if self.input_dim==64:
                latent = torch.Tensor([9.3978e-03, -4.7979e-02,  1.5028e+00, -9.9019e-01,  1.6954e+00,
                                       1.0239e+00,  2.6542e-02,  6.4998e-01,  2.1610e-02,  1.3277e-02])
                latent= torch.transpose(latent.repeat(20, 1), 0, 1)
            else:
                latent = torch.zeros(self.latent_dim, 20)
            latent[i, :] = torch.linspace(-self.latent_range, self.latent_range, 20)
            latent = torch.transpose(latent, 0, 1)
            img_recon = self.model.decode(latent)
            recon.append(img_recon.view(-1, 1, self.input_dim, self.input_dim))

        recon = torch.cat(recon)

        fig, ax = plt.subplots(figsize=(10, 8))
        plt.axis('off')
        plt.rcParams["figure.figsize"] = (20, 3)
        self.show_image(make_grid(recon.data, 20, 8))

        if self.input_dim==28:
            step_size = 36
        elif self.input_dim == 64:
            step_size = 70
        else:
            step_size = 210

        for i in range(0, self.latent_dim, 1):
            plt.text(3, (self.input_dim/2.1) + (i*step_size), str(i), color="red")

        print("\n Latent Space Features:")
        plt.show()

