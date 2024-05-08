import os
import numpy as np
import matplotlib.pyplot as plt

def display(dir_PATH):
    '''This function display a random images from the provided directory path
       :param dir_PATH: the directory path where images are stored'''
    directories=[i for i in os.listdir(dir_PATH) if  os.path.isdir(os.path.join(dir_PATH,i))]
    for dir in directories:
        images = os.listdir(os.path.join(dir_PATH,dir))
        print(f'{len(os.listdir(os.path.join(dir_PATH,dir)))} {dir} tumor')

        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12,10))

        for indx, axis in enumerate(axes.flatten()):
            rnd_indx = np.random.randint(0, len(os.listdir(dir_PATH)))
            # https://matplotlib.org/users/image_tutorial.html
            img = plt.imread(os.path.join(dir_PATH,dir) +'/'+ images[rnd_indx])
            imgplot = axis.imshow(img)
            axis.set_title(images[rnd_indx])
            axis.set_axis_off()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])