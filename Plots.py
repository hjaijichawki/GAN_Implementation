import matplotlib.pyplot as plt
import torch
from models import *
nz=128


def plot_loss (G_losses, D_losses, epoch):
    '''This function plots the generator and discriminator loss during each epoch
       :param G_lossses'''
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss - EPOCH "+ str(epoch))
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def plot_loss_global (G_losses_global, D_losses_global):
    '''This function plots the generator and discriminator loss during the training process'''
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Overall")
    plt.plot(G_losses_global,label="G")
    plt.plot(D_losses_global,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()




