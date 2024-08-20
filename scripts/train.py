import argparse

import torch
from torch import nn

from dataloaders.data_loaders import create_dataloader
from models.torch_models import CNN, LeNet5, MnistModel
from optimizers.optimizer import get_optimizer
from scripts.trainer_class import Trainer
from utils.transforms import transforms


def main(args):
    train_dataloader, test_dataloader = create_dataloader(
        train_path="./data/mnist/train",
        test_path="./data/mnist/test",
        transform=transforms,
        batch_size=args.batch_size,
    )

    # Initialize the model
    if args.model.lower() == "lenet":
        model = LeNet5(num_classes=10)
    elif args.model.lower() == "mnist_model":
        model = MnistModel()
    elif args.model.lower() == "cnn":
        model = CNN()
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    # Define criterion
    criterion = nn.CrossEntropyLoss()

    optimizer = get_optimizer(model, args.optimizer, args.lr)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_dataloader,
        val_loader=test_dataloader,
        num_epochs=args.num_epochs,
        device=device,
    )

    trainer.train()


if __name__ == "__main__":
    # Define command line arguments
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "--model", type=str, required=True, help="Model name (lenet, mnist_model, cnn)"
    )
    parser.add_argument(
        "--optimizer", type=str, default="adam", help="Optimizer (adam, sgd, rmsprop)"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, cuda)")

    args = parser.parse_args()
    main(args)
