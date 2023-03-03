import os
import cv2
from PIL import Image

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets

"""
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
    "test": transforms.Compose([transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
}

class imageLoader(Dataset):

    def __init__(self, image_label, train_or_test=False):
        self.image_label = image_label
        self.train_or_test = train_or_test

    def __getitem__(self, item):
        image, label = self.image_label[item]

        if self.train_or_test:
            image = data_transform["train"](image)
        else:
            image = data_transform["test"](image)

        return image, label

    def __len__(self):
        return len(self.image_label)

class pathLoader(Dataset):

    def __init__(self, image_label_path, arg=False):
        self.image_label_path = image_label_path
        self.arg = arg

    def __getitem__(self, item):
        image_path, label = self.image_label_path[item]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        if self.arg:
            image = data_transform["train"](image)
        else:
            image = data_transform["test"](image)

        return image, label

    def __len__(self):
        return len(self.image_label_path)
        
"""

def getDataLoader(data_dir, batch_size, num_workers, train_or_test=False):

    # dataset = datasets.ImageFolder(root=data_dir, transform=data_transform["train" if train_or_test else "test"])
    if data_dir == "./data/CIFAR10":
        dataset = datasets.CIFAR10(root=data_dir,
                                   train=train_or_test,
                                   transform=transforms.Compose([transforms.RandomResizedCrop(224),
                                                                 transforms.RandomHorizontalFlip(),
                                                                 transforms.ToTensor(),
                                                                 transforms.Normalize([0.5, 0.5, 0.5],
                                                                                      [0.5, 0.5, 0.5])]),
                                   download=True)
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=train_or_test,
                                num_workers=num_workers)
    elif data_dir == "./data/MNIST":
        dataset = datasets.MNIST(root=data_dir,
                                 train=train_or_test,
                                 transform=transforms.Compose([transforms.ToTensor(),
                                                               transforms.Normalize((0.1307,), (0.3081,))]),
                                 download=True)
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=train_or_test,
                                num_workers=num_workers)
    else:
        raise Exception("Error: Can't find data directory name called {}.".format(data_dir))

    return dataloader, dataset
