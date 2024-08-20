|-- data/
|   |-- raw/                  # raw data
|   |-- processed/            # processed data
|
|-- logs/                     # logs folder
|   |-- log.txt
|
|-- losses/                   # losses package initialization
|   |-- __init__.py
|
|-- metrics/                  # metics package initilaztion
|   |-- __init__.py
|
|-- models/
|   |-- __init__.py           # model package initialization
|   |-- model.py              # model definition
|   |-- ...                   # other model files
|
|-- utils/
|   |-- __init__.py           # utilities package initialization
|   |-- utils.py              # utility functions
|   |-- ...                   # other utility files
|
|-- notebooks/
|   |-- exploratory.ipynb     # notebook for data exploration
|   |-- ...                   # other notebooks
|
|-- scripts/
|   |-- train.py              # script to train the model
|   |-- train_class.py        # script to implement training proccess
|       |-- Trainer           # class realizing training and validating model
|       |-- SaveBestModel     # class realizing saving best model in onnx format for faster inference and .pth format
|   |-- inference.py          # script to reliaze inference with onnxruntime
|   |-- ...                   # other scripts
|
|-- config.py                 # configuration file
|-- main.py                   # main script to run the project
|-- requirements.txt          # list of project dependencies
|-- README.md                 # project documentation
|-- .gitignore                # git ignore file


# TO DO:
- Reduce on plateau
- early stopping
- argparse inference - Done
- using calculate_metric() in train_class
- batch_inference
- add graphics of training
