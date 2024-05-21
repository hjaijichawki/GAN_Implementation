import time
import numpy as np 
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from Directory_creation import *
from models import *
from Data_Preparation import *
from Display import *
from Plots import *
from Transformations import *
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils




dataset_path='/content/drive/MyDrive/Brain_Tumor_Detection'
dir_PATH='TRAIN'

create_dir(dir_PATH)
prepare(dataset_path)
dataloader=tranform_images(dir_PATH)
display(dir_PATH)


batch_size = 32
LR_G = 0.0001
LR_D = 0.005

beta1 = 0.5
epochs = 50

real_label = 0.9
fake_label = 0
nz = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

netG = Generator(nz).to(device)
netD = Discriminator().to(device)
criterion = nn.MSELoss()
optimizerD = optim.Adam(netD.parameters(), lr=LR_D, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=LR_G, betas=(beta1, 0.999))

fixed_noise = torch.randn(25, nz, 1, 1, device=device)

G_losses = []
D_losses = []
G_losses_global = []
D_losses_global = []
epoch_time = []

def show_generated_img(n_images=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample = []
    for _ in range(n_images):
        noise = torch.randn(1, nz, 1, 1, device=device)
        gen_image = netG(noise).to("cpu").clone().detach().squeeze(0)
        gen_image = gen_image.numpy().transpose(1, 2, 0)
        sample.append(gen_image)

    figure, axes = plt.subplots(1, len(sample), figsize = (64,64))
    for index, axis in enumerate(axes):
        axis.axis('off')
        image_array = sample[index]
        axis.imshow(image_array)

    plt.show()
    plt.close()

for epoch in range(epochs):

    start = time.time()
    for ii, (real_images, train_labels) in tqdm(enumerate(dataloader), total=len(dataloader)):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_images = real_images.to(device)
        batch_size = real_images.size(0)
        labels = torch.full((batch_size, 1), real_label, device=device)

        output = netD(real_images)
        errD_real = criterion(output, labels)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        labels.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, labels)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        labels.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, labels)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        if (ii+1) % (len(dataloader)//2) == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch + 1, epochs, ii+1, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
    G_losses_global.append(np.mean(G_losses))
    D_losses_global.append(np.mean(D_losses))
    plot_loss (G_losses, D_losses, epoch)
    G_losses = []
    D_losses = []
    show_generated_img()


    epoch_time.append(time.time()- start)

#             valid_image = netG(fixed_noise)



os.makedirs('generated_images')

generate_images('generated_images')
show_generated_images('generated_images')
