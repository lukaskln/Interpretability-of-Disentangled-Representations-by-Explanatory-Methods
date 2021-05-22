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
            if self.input_dim == 64:
                latent = torch.Tensor([1.2204e+00,  6.0715e+00, -1.7371e+00,  3.3546e-02, -6.5440e-03,
                                       2.2066e+00, -1.7470e+00, -9.8487e-04,  2.1421e-02, -2.4422e+00])
                latent = torch.transpose(latent.repeat(20, 1), 0, 1)
            elif self.input_dim > 200:
                if self.latent_dim==10:
                    latent = torch.Tensor([7.4399e-02,  2.9964e-01, -5.9140e+00, -3.9806e+00, -3.6684e-02,
                                        2.7512e-01, -8.1353e+00, -4.7633e-01,  7.2584e-01, -1.9241e+00])
                    latent = torch.transpose(latent.repeat(20, 1), 0, 1)
                else:
                    latent = torch.Tensor([-1.0080, -1.3824, -2.5107, -6.9836, -0.4719, -0.4406,  2.2717, -1.8434,
                                           -1.4958,  2.7429, -3.1879, -0.0596, -1.5097,  2.1459, -0.6683, -4.1047,
                                           -1.6322, -1.4764, -0.3565, 23.0232])
                    latent = torch.transpose(latent.repeat(20, 1), 0, 1)                    
            else:
                latent = torch.ones(self.latent_dim, 20)
            latent[i, :] = torch.linspace(-self.latent_range, self.latent_range, 20)
            latent = torch.transpose(latent, 0, 1)
            img_recon = self.model.decode(latent)
            recon.append(img_recon.view(-1, 1, self.input_dim, self.input_dim))

        recon = torch.cat(recon)

        fig, ax = plt.subplots()
        plt.figure(figsize=(15, 15), dpi=200)
        plt.axis('off')
        self.show_image(make_grid(recon.data, 20, 8))

        if self.input_dim == 28:
            step_size = 36
        elif self.input_dim == 64:
            step_size = 70
        else:
            step_size = 320

        for i in range(0, self.latent_dim, 1):
            plt.text(3, (self.input_dim/2.1) + (i*step_size), str(i), color="red")

        print("\n Latent Space Features:")
        plt.savefig('./images/latent space features.png')

