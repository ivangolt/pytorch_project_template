"""Dataloaders module"""

import os

from torch.utils.data import Dataset
from torchvision.io import read_image


class MnistDataset(Dataset):
    """Class for loading dataset from path with using torch transforms

    Args:
        Dataset (_type_): _description_
    """

    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        self.image_files = []
        self.labels = []

        for filename in os.listdir(path=path):
            if filename.endswith(".png"):
                label = filename[0]
                self.image_files.append(filename)
                self.labels.append(int(label))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = os.path.join(self.path, self.image_files[index])
        image = read_image(image_path)
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return image, label
