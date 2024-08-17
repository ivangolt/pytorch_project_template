import argparse

import torch
import torch.optim as optim
from torch import nn

from dataloaders.data_loaders import create_dataloader
from models.torch_models import LeNet5, MnistModel
from scrpits.trainer_class import Trainer
from utils.transforms import transforms

# Define command line arguments
parser = argparse.ArgumentParser(description="Train a model")
parser.add_argument("--model", type=str, help="Model name (model1, model2)")
parser.add_argument(
    "--optimizer", type=str, default="adam", help="Optimizer (adam, sgd, rmsprop)"
)
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, cuda)")
args = parser.parse_args()


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
else:
    raise ValueError(f"Unsupported model: {args.model}")

# Define criterion
criterion = nn.CrossEntropyLoss()


# Define optimizer based on command line argument
if args.optimizer.lower() == "adam":
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer.lower() == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
elif args.optimizer.lower() == "rmsprop":
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
else:
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")

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
