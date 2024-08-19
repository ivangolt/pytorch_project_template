import argparse

import numpy as np
import onnxruntime as ort

from utils.preprocess_input_image import preprocess_image

# onnx_model_path = "./outputs/models/LeNet5.onnx"

# image_path = "C:/VSCode_projects/pytorch_project_template/data/mnist/test/7_000638.png"

parser = argparse.ArgumentParser(description="model inference")
parser.add_argument("--model", type=str, help="Model name (model1, model2)")
parser.add_argument(
    "--path_of_image", type=str, default=None, help="Path for image to inference"
)
args = parser.parse_args()

if args.model.lower() == "lenet":
    model = "./outputs/models/LeNet5.onnx"
# elif args.model.lower() == "mnist_model":
#     model = MnistModel()
# elif args.model.lower() == "cnn":
#     model = CNN()
else:
    raise ValueError(f"Unsupported model: {args.model}")


def onnx_inference(image_path: str, model_path: str):
    """Function for onnx inference

    Args:
        image_path (str): path to image
        model_path (str): path to model

    Returns:
        int: output label
    """

    input_data = preprocess_image(image_path)
    ort_session = ort.InferenceSession(model_path)

    # Perform inference
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name

    # Run the model on the input data
    output = ort_session.run([output_name], {input_name: input_data})
    # Process the output (for example, softmax for classification)
    output_tensor = np.array(output[0])

    return f"Predicted label: {output_tensor.argmax()}"


print(onnx_inference(image_path=args.path_of_image, model_path=args.model))
