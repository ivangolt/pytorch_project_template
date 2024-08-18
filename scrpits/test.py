import matplotlib.pyplot as plt
import torch
from torchinfo import summary
from torchvision.transforms import v2

from dataloaders.data_loaders import create_dataloader
from models.torch_models import MnistModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = MnistModel()


print(
    summary(
        model=model,
        input_size=(1, 1, 28, 28),
        verbose=0,
        col_names=["input_size", "output_size", "num_params", "trainable"],
    )
)


transforms = v2.Compose(
    [
        v2.Resize(size=(224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.5,), std=(0.5,)),
    ]
)

train_dataloader, test_dataloader = create_dataloader(
    train_path="./data/mnist/train",
    test_path="./data/mnist/test",
    transform=transforms,
    batch_size=1,
)
# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
