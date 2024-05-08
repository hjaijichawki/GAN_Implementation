import os
import shutil


def prepare(dataset_path):
    '''This function splits images from the original dataset path into the training folder TRAIN
       :param dataset_path: the directory path where the original images are stored'''
    ignored={"pred"}
    directories=[i for i in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path,i))]
    #split data by train/test/val
    for CLASS in directories:
        if CLASS not in ignored:
            if not CLASS.startswith('.'):
                for (n,FILE_NAME) in enumerate(os.listdir(dataset_path+'/'+CLASS)):
                    img=dataset_path+'/{}/{}'.format(CLASS,FILE_NAME)
                    shutil.copy(img,'TRAIN/{}/{}'.format(CLASS.upper(),FILE_NAME))