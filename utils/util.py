"""Utils helpers script"""

import matplotlib.pyplot as plt

plt.style.use("ggplot")


def save_plots(train_acc, valid_acc, train_loss, valid_loss):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_acc, color="green", linestyle="-", label="train accuracy")
    plt.plot(valid_acc, color="blue", linestyle="-", label="validataion accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("./results/accuracy.png")

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color="orange", linestyle="-", label="train loss")
    plt.plot(valid_loss, color="red", linestyle="-", label="validataion loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("./results/loss.png")
