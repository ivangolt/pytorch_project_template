"""Module with functions of metrics"""

import torch


def calculate_accuracy(outputs, labels) -> float:
    """_summary_

    Args:
        outputs (_type_): predicted outputs
        labels (_type_): true label

    Returns:
        float: return accuracy
    """

    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    accuracy = correct / total * 100
    return accuracy
