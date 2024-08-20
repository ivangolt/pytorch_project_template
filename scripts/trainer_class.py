import logging
import os

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


class Trainer:
    """Class for training process"""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 10,
        device: str = "cpu",
    ):
        self.device = device
        self.model = model
        self.model = self.model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs

        self.criterion = criterion
        self.optimizer = optimizer

        # Initialize lists for storing loss and accuracy values
        self.train_losses = []
        self.train_accs = []
        self.val_accs = []
        self.val_losses = []

        self.save_best_model = SaveBestModel(
            name_of_model=model.__class__.__name__, device=self.device
        )

    def train(self, save_model=True):
        logging.info(f"Starting traing on: {self.device}")

        scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.1, patience=5, verbose=True
        )

        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            progress_bar = tqdm(
                enumerate(self.train_loader),
                total=len(self.train_loader),
                desc=f"Epoch : {epoch+1}/{self.num_epochs}",
            )
            for i, (inputs, labels) in progress_bar:
                # Zero the parameter gradients
                self.optimizer.zero_grad()

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Update progress bar
                progress_bar.set_postfix(
                    Loss=f"{round(loss.item(),4)}",
                    Accuracy=f"{round((100 * correct / total),3)}",
                )

            # Store and print average loss and accuracy per epoch
            epoch_loss = running_loss / len(self.train_loader)
            epoch_acc = 100 * correct / total
            self.train_losses.append(epoch_loss)
            self.train_accs.append(epoch_acc)

            scheduler.step(epoch_loss)

            # Log the current learning rate
            lr_rate = self.optimizer.param_groups[0]["lr"]
            logging.info(
                f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%, Learning Rate: {lr_rate:.6f}"
            )

            # Validate the model
            val_acc, val_loss = self.validate()
            self.val_accs.append(val_acc)
            self.val_losses.append(val_loss)

            # saving the model
            if save_model:
                self.save_best_model(
                    current_valid_loss=val_loss,
                    model=self.model,
                    epoch=epoch,
                    optimizer=self.optimizer,
                    criterion=self.criterion,
                )

    def validate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.inference_mode():
            running_loss = 0.0
            for inputs, labels in self.val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        loss = running_loss / len(self.val_loader)
        accuracy = 100 * correct / total

        print(
            f"Epoch :  Validation Accuracy on this epoch: {accuracy}% and Loss: {loss}"
        )
        return accuracy, loss


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.
    """

    def __init__(
        self,
        best_valid_loss=float("inf"),
        name_of_model: str = None,
        device: str = "cpu",
    ):
        self.best_valid_loss = best_valid_loss
        self.name_of_model = name_of_model
        self.device = device
        os.makedirs("outputs", exist_ok=True)

    def __call__(self, current_valid_loss, epoch, model, optimizer, criterion):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            logging.info(f"Best validation loss: {self.best_valid_loss}")
            logging.info(f"Saving best model for epoch: {epoch+1}/n")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": criterion,
                },
                f=f"./outputs/models/{self.name_of_model}.pth",
            )
            # Create a dummy input tensor with the correct shape
            dummy_input = torch.randn(1, 1, 28, 28).to(self.device)

            # Export the model to ONNX format
            torch.onnx.export(
                model,  # model being run
                dummy_input,  # model input (or a tuple for multiple inputs)
                f=f"./outputs/models/{self.name_of_model}.onnx",  # where to save the model (can be a file or file-like object)
                export_params=True,  # store the trained parameter weights inside the model file
                opset_version=11,  # the ONNX version to export the model to
                do_constant_folding=True,  # whether to execute constant folding for optimization
                input_names=["input"],  # the model's input names
                output_names=["output"],  # the model's output names
                dynamic_axes={
                    "input": {0: "batch_size"},  # variable length axes
                    "output": {0: "batch_size"},
                },
            )
            logging.info("Save best model in onnx format")
