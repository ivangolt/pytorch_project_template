"""Dataloaders module"""
import os
from torchvision.io import read_image

class MnistDataset(Dataset):

    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        self.image_files = []
        self.labels = []


    
        for filename in os.listdir(path=path):
            if filename.endswith(".png"):
                label = filename[0]
                self.image_files.append(filename)
                self.labels.append(label)


    def __len__(self):
        return(len(self.image_files))

    def __getitem__(self, index):

        image_path = os.path.join(self.path, self.image_files[index])
        image = read_image(image_path)
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return image, label

