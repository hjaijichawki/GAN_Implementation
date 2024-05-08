import matplotlib.pyplot as plt
import os 
import numpy as np


def show_generated_images(dir_PATH):
    '''This functions display the generated images from the saving directory
       :param dir_PATH: the path where the generated images are saved'''
    images = os.listdir(dir_PATH)
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12,10))
    for indx, axis in enumerate(axes.flatten()):
        rnd_indx = np.random.randint(0, len(os.listdir(dir_PATH)))
        # https://matplotlib.org/users/image_tutorial.html
        img = plt.imread(dir_PATH +'/'+ images[rnd_indx])
        imgplot = axis.imshow(img)
        axis.set_axis_off()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])