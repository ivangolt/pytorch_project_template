"""create dataloader using template Dataset"""
from dataloaders.dataset import MnistDataset
import sys
import pandas


def create_dataloader(train_path: str, 
                      test_path: str,
                      transform: transforms.Compose,
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
    A tuple of (train_dataloader, test_dataloader, class_names).
    Where class_names is a list of the target classes.
    Example usage:
      train_dataloader, test_dataloader, class_names = \
        = create_dataloaders(train_dir=path/to/train_dir,
                             test_dir=path/to/test_dir,
                             transform=some_transform,
                             batch_size=32,
                             num_workers=4)
    """

    