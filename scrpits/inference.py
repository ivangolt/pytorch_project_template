import numpy as np
import onnxruntime as ort

from utils.preprocess_input_image import preprocess_image

onnx_model_path = "./outputs/models/LeNet5.onnx"

image_path = "C:/VSCode_projects/pytorch_project_template/data/mnist/test/7_000638.png"


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


print(onnx_inference(image_path=image_path, model_path=onnx_model_path))
