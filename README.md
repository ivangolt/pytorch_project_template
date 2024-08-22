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


# Run train a model
```python .\scripts\train.py --model lenet --optimizer adam --batch_size 64 --lr 0.01 --num_epochs 1 --device cuda```

# Run inference a model
```python .\scripts\inference.py --model_name lenet --image_path "image path"```

# TO DO:
- Reduce on plateau - Done
- early stopping - Done
- argparse inference - Done
- using calculate_metric() in train_class - Done
- add graphics of training
