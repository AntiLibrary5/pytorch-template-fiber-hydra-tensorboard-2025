# Deep Learning Project Template
[![python](https://img.shields.io/badge/-Python_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![tensorboard](https://img.shields.io/badge/Logging-TensorBoard-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/tensorboard)
[![loguru](https://img.shields.io/badge/Logging-Loguru_0.7-blue?logo=python&logoColor=white)](https://loguru.readthedocs.io/)

A lightweight template for **PyTorch** based deep learning projects with main features of configuration management (**Hydra**), 
logging (**Loguru**+**TensorBoard**), and hardware-agnostic training (**Lightning Fiber**). Designed for rapid experimentation while enforcing best practices.

![GitHub Repo stars](https://img.shields.io/github/stars/AntiLibrary5/pytorch-template-fiber-hydra-tensorboard-2025?style=social)
![GitHub forks](https://img.shields.io/github/forks/AntiLibrary5/pytorch-template-fiber-hydra-tensorboard-2025?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/AntiLibrary5/pytorch-template-fiber-hydra-tensorboard-2025?style=social)

## Table of Contents
- [Why This Template?](#why-this-template)
- [How to use](#how-to-use)
  - [Project Structure](#project-structure)
  - [Environment Setup](#environment-setup)
  - [Training](#training)
  - [Inference](#inference)
- [Configuration Management](#configuration-management)
  - [Core Concepts](#core-concepts)
  - [Creating New Configurations](#creating-new-configurations)
- [Logging and Visualization](#logging-and-visualization)
  - [Message Logging](#message-logging)
  - [Tensorboard Logging](#tensorboard-logging)
- [Toy Example](#toy-example-image-reconstruction)
- [Extending the Template](#extending-the-template)
- [License](#license)

## Why This Template?
> **Note**: This section provides detailed background and motivation. If you're just looking to get started quickly, you can skip to the [Quick Start](#quick-start) section. If you're interested in why this template exists and what problems it solves, expand below.

<details>
<summary>🤔 Click to expand the motivation and background</summary>

There are plenty of deep learning templates out there—so why this one?  

As a research engineer in computer vision for over four years, my workflow has consistently involved:  
- Reviewing SOTA papers  
- Implementing papers or adapting their existing codebases to specific datasets and problems  

However, existing implementations often come with excessive complexity. Each codebase has a different structure, making it time-consuming to adapt. In reality, I only need the essentials:  
- **Dataset processing**  
- **Model architecture**  
- **Loss functions**  
- **Training logic**  
- ...

Everything else should be familiar and easy to modify for experiments. I often end up stripping down implementations to the bare minimum and rewriting them for:  
- **Configurable experiment management**  
- **Effective logging (preferably free)**  
- **Seamless multi-GPU support**  

---

### Why Not Use Existing Templates?  

Yes, there are other well-structured templates, but:  
- **They are over-engineered** → Hard to modify, too much boilerplate  
- **They impose strict frameworks** → Require learning Lightning or other abstractions  
- **They add unnecessary complexity** → I just need a simple, adaptable structure  

What I need is simple:  
✅ **Run multiple experiments with different models & settings**  
✅ **Quickly switch configurations**  
✅ **Track experiments efficiently**  
✅ **Easy inference with saved settings**  
✅ **Monitor training behavior with intuitisve logging**  
✅ **No CUDA/CPU/hardware headaches**  

---

### What This Template Offers  

- **Configuration Management** → Hierarchical configs with Hydra  
- **Experiment Tracking** → Auto-save & load experiment settings  
- **Logging** → Console & file logging with Loguru  
- **Visualization** → TensorBoard support for metrics, images, and models  
- **Hardware Agnostic** → Lightning Fabric (better flexibility than PyTorch Lightning)  
- **Lean & Adaptable** → No unnecessary overhead, quick to modify  

This template is designed to **keep things simple, flexible, and experiment-focused**—without unnecessary complexity.  

</details>

# How to use
Use this mainly for the config management and logging features. A toy example of a reconstruction autoencoder for a random image
to show it works and show where the dataset, model, loss optimizers, training/validation/inference logic, vis, io, other utils could go.
You're in control.

### Project Structure
```
├── configs/                # Configuration files
│   ├── config.yaml         # Main configuration
│   ├── data/               # Dataset configurations
│   ├── experiment/         # Experiment configurations
│   ├── model/              # Model configurations
│   └── training/           # Training configurations
├── model_save/             # Saved models and experiment data
├── src/                    # Source code
│   ├── data/               # Dataset implementations
│   ├── model/              # Model implementations
│   ├── utils/              # Utility functions
│   │   ├── logging/        # Logging utilities
│   │   │   ├── msg_logger.py  # Message logging with Loguru
│   │   │   └── tb_logger.py   # TensorBoard logging
│   │   ├── io.py          # I/O utilities
│   │   ├── utils.py       # General utilities
│   │   └── visualization.py # Visualization utilities
│   ├── infer.py           # Inference script
│   └── train.py           # Training script
└── requirements.txt        # Dependencies
```

### Environment Setup

Set up your own environment, but to you need atleast mentioned in the provided `requirements.txt` file to use the features here and
run the toy example (adapt the cuda version or just use your own installation proc):

```bash
conda create -n minimal python=3.10
conda activate minimal
pip install --force-reinstall -r requirements.txt
```

### Training

To train a model with the default configuration:

```bash
python src/train.py
```

This will:
1. Create an experiment directory in `model_save/` with the experiment name `<exp_name>`
2. Save the configuration used for training in `model_save/<exp_name>/.hydra` (for reference and used for resuming exp or inference)
3. Log messages to both console and a log file `model_save/<exp_name>/train.log` in the experiment directory 
4. Log metrics and visualizations to TensorBoard in `model_save/<exp_name>/tb`

**Key CLI Overrides**:
You can override any configuration parameter from the command line:
```bash
# Change run params 
python src/train.py experiment.exp_name=my_experiment training.epochs=100 training.batch_size=64

# Change model architecture
python src/train.py model=complex

# Multi-GPU training
python src/train.py training.devices=2 training.accelerator="gpu"

# Mixed precision
python src/train.py training.precision="16-mixed"
```

### Inference
```bash
python src/infer.py --experiment <exp_name>
```
- Loads config from original training run
- Saves predictions to `model_save/<exp_name>/preds`

Optionally, you can also use CLI to over-ride params during inference (needed sometimes)
```bash
python src/infer.py --experiment <exp_name> data.image_type=tif
```

## Configuration Management

### Core Concepts
1. **Hierarchical Configs**  
   Compose configurations from multiple files:
   ```yaml
   # configs/config.yaml
   defaults:
     - experiment: default
     - training: default
     - model: base
     - data: default
   ```

2. **Experiment-Specific Settings**  
   ```yaml
   # configs/experiment/default.yaml
   exp_name: "unet_baseline"
   description: "Base UNet with MSE loss"
   ```

3. **Model Zoo**  
   Switch architectures via config:
   ```yaml
   # configs/model/unet.yaml
   _target_: src.model.unet.UNet
   in_channels: 1
   out_channels: 1
   initial_features: 64
   ```

### Creating New Configurations

To create a new configuration, add a YAML file to the appropriate directory:

1. For a new model: `configs/model/my_model.yaml`
2. For a new dataset: `configs/data/my_dataset.yaml`
3. For a new training setup: `configs/training/my_training.yaml`

Then update the default yaml
   ```yaml
   # configs/config.yaml
   defaults:
     - experiment: my_experiment
     - training: my_training
     - model: my_model
     - data: my_dataset
   ```

OR use it from CLI with:
```bash
python src/train.py model=my_model data=my_dataset training=my_training experiment=my_experiment
```

## Logging and Visualization

### Message Logging

The template uses Loguru for message logging. The flow is simple. Look at `src/train.py`. Here is a snippet, in your
entry point:
```python
import hydra
from src.utils.logging.msg_logger import setup_logging
# Set up msg logger
hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
setup_logging(exp_dir=hydra_cfg.run.dir, log_filename=hydra_cfg.job.name)
```
Then in any file/module/function/class, do 
```python
from loguru import logger
epoch=0
loss=1
logger.info(f"Epoch {epoch} loss: {loss:.4f}")
logger.info("Starting training process")
logger.warning("GPU memory is running low")
logger.error("Failed to load dataset")
something = `something`
logger.opt(colors=True).info("<blue>Using color to highlight(s):</blue> <green>{}</green>", something)
```
Logs are saved to both the console and a log file in the experiment directory.

### TensorBoard Logging

The template includes a TensorBoard logger for visualizing metrics and images:

```python
from src.utils.logging.tb_logger import TensorBoardLogger
tb_logger = TensorBoardLogger(tb_dir)

# Log a scalar value
tb_logger.log_scalar("training/loss", loss_value, step)

# Log images
tb_logger.log_images("training/generated_images", sample_images, step)

# Log model graph
tb_logger.log_model_graph(model, dummy_input)
```

To view TensorBoard logs:

```bash
tensorboard --logdir model_save/my_experiment/tb
```

## Toy Example: Image Reconstruction

**Implemented Components**:
- UNet architecture with configurable depth/features
- Random image dataset (template for easy replacement)
- MSE loss autoencoder training
- Reconstruction visualization utilities

![toy.png](assets/toy.png)

## Extending the Template

This template is designed to be extended for your specific needs:

### 1. Add New Models
1. Implement model in `src/model/`
2. Create config in `configs/model/`
3. Update main config:
   ```yaml
   defaults:
     - model: your_model  # in configs/config.yaml
   ```

### 2. Add Datasets
1. Implement dataset class in `src/data/`
2. Update data config:
   ```yaml
   # configs/data/default.yaml
   data:
     name: "your_dataset"
     image_size: 256
   ```

### 3. Modify Training Logic
1. Edit `src/train.py` core loop
2. Add new metrics/visualizations
3. Extend logging as needed

### 4. Add custom metrics and visualizations using the TensorBoard logger

## Again

- Make use of the core features of config management and logging for deep learning exps
- Look at the lightweight toy example for what the workflow could be
- Easy (relatively easy/easier) to open it and adapt for your use case (hopefully)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=AntiLibrary5/pytorch-template-fiber-hydra-tensorboard-2025&type=Date)](https://www.star-history.com/#AntiLibrary5/pytorch-template-fiber-hydra-tensorboard-2025&Date)


## License
```
MIT License

Copyright (c) 2021 ashleve

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```