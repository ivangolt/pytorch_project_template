import argparse

import numpy as np
import onnxruntime as ort

from utils.preprocess_input_image import preprocess_image


def onnx_inference(image_path: str, model_path: str):
    """Function for ONNX inference

    Args:
        image_path (str): path to image
        model_path (str): path to model

    Returns:
        str: output label
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


def main():
    parser = argparse.ArgumentParser(description="ONNX Model Inference")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model (e.g., 'lenet')",
    )
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to the input image"
    )

    args = parser.parse_args()

    # Map model names to their corresponding file paths
    model_paths = {
        "lenet": "./outputs/models/LeNet5.onnx",
        "resnet": "./outputs/models/ResNet.onnx",
        "mobilenet": "./outputs/models/MobileNet.onnx",
        # Add more models as needed
    }

    model_path = model_paths.get(args.model_name.lower())

    if model_path is None:
        raise ValueError(
            f"Model '{args.model_name}' not recognized. Available models: {list(model_paths.keys())}"
        )

    result = onnx_inference(image_path=args.image_path, model_path=model_path)
    print(result)


if __name__ == "__main__":
    main()
