"""this script consist transforms to input image"""

import torch
from torchvision.transforms import v2

transforms = v2.Compose(
    [
        v2.Resize(size=(224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.5,), std=(0.5,)),
    ]
)
