import torch
from torchvision import datasets, transforms
import torch.utils.data

def tranform_images(PATH):
    '''This function applies a set of images transformations on the images existing in the provided path
       :param PATH: the directory path where images are stored'''
    batch_size = 32
    image_size = 32

    random_transforms = [transforms.ColorJitter(), transforms.RandomRotation(degrees=20)]
    transform = transforms.Compose([transforms.Resize(64),
                                    transforms.CenterCrop(64),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomApply(random_transforms, p=0.2),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_data = datasets.ImageFolder(PATH, transform=transform)
    dataloader = torch.utils.data.DataLoader(train_data, shuffle=True,
                                            batch_size=batch_size)
    return dataloader