# Semantic Segmentation Training Framework

A PyTorch framework for training semantic segmentation models. This repo gives you a single interface to work with eight different segmentation architectures, handling the optimization and training specifics for each one.

## Supported Models

- **UNet** - The go-to encoder-decoder for medical imaging
- **FCN ResNet50/101** - Fully convolutional networks built on ResNet
- **DeepLabV3 ResNet50/101** - Uses atrous convolutions with ResNet backbones
- **DeepLabV3 MobileNet** - Smaller DeepLabV3 that runs on mobile devices
- **LRASPP MobileNet** - Stripped-down version for speed
- **SegNet** - Uses pooling indices to upsample efficiently

## What You Get

- Single pipeline for all eight architectures
- Transfer learning with pre-trained weights
- Separate log files for each component (trust me, this helps)
- Configure everything from the command line
- Auto-saves checkpoints and runs validation
- Handles any image size and custom datasets
- Multi-GPU support if you've got the hardware

## Setup

Install the dependencies:

```bash
pip install torch torchvision numpy
```

## How to Use

### Quick Start

```bash
python main.py \
  --module unet \
  --classes 21 \
  --training-path ./data/train \
  --validation-path ./data/val \
  --testing-path ./data/test \
  --output ./model.pt
```

### Command Line Arguments

#### Model Setup
- `--module` - Which architecture to use (required)
- `--classes` - Number of segmentation classes (required)
- `--channels` - Input channels, typically 3 for RGB
- `--weights` - Use pre-trained weights (enabled by default)

#### Data Paths
- `--training-path` - Where your training data lives (required)
- `--validation-path` - Where your validation data lives (required)
- `--testing-path` - Where your test data lives (required)
- `--weights-path` - Path to checkpoint if resuming training

#### Training Parameters
- `--epochs` - How long to train (default: 50)
- `--batch-size` - Samples per batch (default: 16)
- `--learning-rate` - Initial learning rate (default: 0.001)
- `--dimensions` - Image size as height width (default: 512 512)
- `--workers` - Number of data loading threads (default: 4)
- `--seed` - Random seed for reproducibility

#### Advanced Options
- `--weight-decay` - L2 regularization strength (default: 0.0005)
- `--gamma` - Learning rate decay factor (default: 0.1)
- `--parallelism` - Use multiple GPUs (default: False)
- `--output` - Where to save the trained model (default: model.pt)

## Logging

Logs get written to four separate files:

- `main.log` - High-level program flow
- `loader.log` - Data loading operations
- `modules.log` - Model-specific operations
- `trainer.log` - Training progress and metrics

## Structure

```
.
├── main.py           # Entry point
├── trainer.py        # Training logic
├── modules.py        # All model implementations
└── logs/            # Where logs end up
```

## Requirements

- Python 3.7+
- PyTorch 1.8+
- torchvision
- NumPy

## Notes

Pre-trained weights load automatically when you have them. They're especially useful when you're working with smaller datasets.

Segmentation eats up memory fast. If you hit OOM errors, drop the batch size or shrink your image dimensions.

The defaults work well enough to get started, but you'll want to tune them based on your specific dataset and hardware.

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.
