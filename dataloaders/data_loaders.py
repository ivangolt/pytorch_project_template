"""create dataloader using template Dataset"""

import logging

from torch.utils.data import DataLoader
from torchvision.transforms import v2

from dataloaders.dataset import MnistDataset

logging.basicConfig(level=logging.INFO)


def create_dataloader(
    train_path: str,
    test_path: str,
    transform: v2.Compose,
    batch_size: int,
):
    """Creates training and testing DataLoaders.

     Takes in a training directory and testing directory path and turns
    them into PyTorch Datasets and then into PyTorch DataLoaders.

    Args:
    train_dir: Path to training directory.
    test_dir: Path to testing directory.
    transform: torchvision transforms to perform on training and testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.

    Returns:
    A tuple of (train_dataloader, test_dataloader).
    Where class_names is a list of the target classes.
    Example usage:
      train_dataloader, test_dataloader = /
        = create_dataloaders(train_dir=path/to/train_dir,
                             test_dir=path/to/test_dir,
                             transform=some_transform,
                             batch_size=32,
                             )
    """

    logging.info("Starting load train dataset from 'train path'")
    train_data = MnistDataset(path=train_path, transform=transform)
    logging.info("Load train dataset from 'train path'")

    logging.info("Starting load test dataset from 'test path'")
    test_data = MnistDataset(path=test_path, transform=transform)
    logging.info("Load train dataset from 'test path'")

    train_dataloader = DataLoader(
        dataset=train_data, batch_size=batch_size, shuffle=True
    )
    test_dataloader = DataLoader(
        dataset=test_data, batch_size=batch_size, shuffle=False
    )

    return train_dataloader, test_dataloader
