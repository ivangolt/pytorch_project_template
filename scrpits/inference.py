import numpy as np
import onnxruntime as ort

from utils.preprocess_input_image import preprocess_image

onnx_model_path = "./outputs/models/LeNet5.onnx"
ort_session = ort.InferenceSession(onnx_model_path)

image_path = "C:/VSCode_projects/pytorch_project_template/data/mnist/test/7_000638.png"
input_data = preprocess_image(image_path=image_path)

# Perform inference
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name


# Run the model on the input data
output = ort_session.run([output_name], {input_name: input_data})


# Process the output (for example, softmax for classification)
output_tensor = np.array(output[0])

# Print the output (for example, probabilities or logits)
print("Model output:", output_tensor.argmax())
