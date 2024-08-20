import torch.optim as optim


def get_optimizer(model, optimizer_name, lr):
    """Returns the optimizer based on the given name."""
    optimizer_name = optimizer_name.lower()
    if optimizer_name == "adam":
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "sgd":
        return optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == "rmsprop":
        return optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
