import os
import cv2
from PIL import Image

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets

def getTrainDataLoader(data_dir, batch_size, num_workers):
    if data_dir == "./data/CIFAR10":
        dataset = datasets.CIFAR10(root=data_dir,
                                   train=True,
                                   transform=transforms.Compose([transforms.RandomResizedCrop(224),
                                                                 transforms.RandomHorizontalFlip(),
                                                                 transforms.ToTensor(),
                                                                 transforms.Normalize([0.494, 0.485, 0.450],
                                                                                      [0.201, 0.199, 0.202])]),
                                   download=True)
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers)
    elif data_dir == "./data/CIFAR100":
        dataset = datasets.CIFAR100(root=data_dir,
                                    train=True,
                                    transform=transforms.Compose([transforms.RandomResizedCrop(224),
                                                                  transforms.RandomHorizontalFlip(),
                                                                  transforms.ToTensor(),
                                                                  transforms.Normalize([0.507, 0.487, 0.441],
                                                                                       [0.267, 0.256, 0.276])]),
                                    download=True)
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers)
    elif data_dir == "./data/ImageNet":
        # ILSVRC2012 is available locally.
        dataset = datasets.ImageFolder(root='/home/sdc1/dataset/ILSVRC2012/images/train',
                                       transform=transforms.Compose([transforms.RandomResizedCrop(224),
                                                                     transforms.RandomHorizontalFlip(),
                                                                     transforms.ToTensor(),
                                                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                                                          [0.229, 0.224, 0.225])]))
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers)
    else:
        raise Exception("Error: Can't find data directory name called {}.".format(data_dir))

    return dataloader, dataset

def getTestDataLoader(data_dir, batch_size, num_workers):
    if data_dir == "./data/CIFAR10":
        dataset = datasets.CIFAR10(root=data_dir,
                                   train=False,
                                   transform=transforms.Compose([transforms.Resize(256),
                                                                 transforms.CenterCrop(224),
                                                                 transforms.ToTensor(),
                                                                 transforms.Normalize([0.494, 0.485, 0.450],
                                                                                      [0.201, 0.199, 0.202])]),
                                   download=True)
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers)
    elif data_dir == "./data/CIFAR100":
        dataset = datasets.CIFAR100(root=data_dir,
                                    train=False,
                                    transform=transforms.Compose([transforms.Resize(256),
                                                                  transforms.CenterCrop(224),
                                                                  transforms.ToTensor(),
                                                                  transforms.Normalize([0.507, 0.487, 0.441],
                                                                                       [0.267, 0.256, 0.276])]),
                                    download=True)
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers)
    elif data_dir == "./data/ImageNet":
        # ILSVRC2012 is available locally.
        dataset = datasets.ImageFolder(root='/home/sdc1/dataset/ILSVRC2012/images/val',
                                       transform=transforms.Compose([transforms.Resize(256),
                                                                     transforms.CenterCrop(224),
                                                                     transforms.ToTensor(),
                                                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                                                          [0.229, 0.224, 0.225])]))
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers)
    else:
        raise Exception("Error: Can't find data directory name called {}.".format(data_dir))

    return dataloader, dataset
