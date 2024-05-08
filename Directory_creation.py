import os

def create_dir(dir_PATH):
    '''This function creates a directory with the provided path and a subdirectories YES and NO in it.
       :param dir_PATH: the path of the directory that you want to create'''
    os.makedirs(dir_PATH)
    os.makedirs(os.path.join(dir_PATH,'NO'))
    os.makedirs(os.path.join(dir_PATH,'YES'))