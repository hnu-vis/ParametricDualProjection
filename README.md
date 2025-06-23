# Parametric Dual Projection Method

## Installation

The code has been tested with **Python 3.8** on **Ubuntu**. To set up the environment and install dependencies, run:

```bash
pip install -r requirements.txt
```

## Project Structure

The project is organized as follows:

```
.
├── data                    # Data loading modules
│   ├── __init__.py
│   ├── cifar_loader.py
│   └── mnist_loader.py
├── database                # Dataset storage
│   ├── MNIST
│   └── cifar
├── loss                    # Loss functions
│   └── __init__.py
├── model                   # Model definitions
│   ├── __init__.py
│   ├── base_embedder.py    # Base embedding module
│   ├── configs             # Model configurations
│   │   ├── __init__.py
│   │   └── base_config.py  # Base configuration
│   └── modules             # Model submodules
│       └── __init__.py
├── train                   # Training scripts
│   ├── __init__.py
│   └── trainer.py          # Training logic
├── utils                   # Utility functions
│   └── __init__.py
├── train.py                # Main training script
└── metric.py               # Evaluation metrics
```

## Running the Demo

The repository includes pre-configured MNIST and CIFAR-10 datasets located in the `database/` directory.

1. **Train the model**:
   ```bash
   python train.py
   ```

2. **Compute evaluation metrics**:
   ```bash
   python metric.py
   ```

Training logs and metric results are saved to the `log/` directory.

## Configuration

To customize training settings, modify the configuration files in `model/configs/`. These inherit from `model.configs.base_config.BaseConfig`. Update parameters such as learning rate, batch size, or model architecture as needed.

## Using Custom Datasets

To use a custom dataset, ensure it is formatted correctly and compatible with the data loaders in `data/`. Update the loader scripts (`cifar_loader.py` or `mnist_loader.py`) to handle your dataset or create a new loader module.

## Contact

For questions or feedback, please:
- Email: [telosaletheia@gmail.com](mailto:telosaletheia@gmail.com)
- Open an issue in the repository's Issues section.
