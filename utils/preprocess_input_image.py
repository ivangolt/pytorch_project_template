from PIL import Image

from utils.transforms import transforms


def preprocess_image(image_path: str):
    """Function to preprocess input image for inference

    Args:
        image_path (str): path to image to inference

    Returns:
        _type_: numpy tensor
    """

    image = Image.open(image_path)

    input_tensor = transforms(image).unsqueeze(0)

    return input_tensor.numpy()
