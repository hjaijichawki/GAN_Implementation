from tqdm import tqdm
import os
import torch
import torchvision.utils
from models import *


def generate_images(saving_dir,n_images=100, im_batch_size=32, nz=128):
    '''This function generates images with the trained GAN and save it in the provided path
       :param saving_dir: the path where you want to save the generated images
       :param n_images: number of images to generate
       :param im_batch_size: image batch size
       :param nz: the noise'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG = Generator(nz).to(device)
    for i_batch in tqdm(range(0, n_images, im_batch_size)):
        gen_z = torch.randn(im_batch_size, nz, 1, 1, device=device)
        gen_images = netG(gen_z)
        images = gen_images.to("cpu").clone().detach()
        images = images.numpy().transpose(0, 2, 3, 1)
        for i_image in range(gen_images.size(0)):
            torchvision.utils.save_image(gen_images[i_image, :, :, :], os.path.join(saving_dir, f'image_{i_batch+i_image:05d}.png'))