import os
import cv2
from PIL import Image

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
    # TODO: test 部分暂时没写
    "test": transforms.Compose()
}

class imageLoader(Dataset):

    def __init__(self, image_label, arg=False):
        self.image_label = image_label
        self.arg = arg

    def __getitem__(self, item):
        image, label = self.image_label[item]

        if self.arg:
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

def getDataLoader(data_dir, batch_size, num_workers, arg=False):

    dataset = datasets.ImageFolder(root=data_dir,
                                   transform=data_transform["train" if arg else "test"])
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=arg,
                            num_workers=num_workers)

    return dataloader, dataset
