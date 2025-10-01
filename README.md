# Semantic Segmentation Training Framework

A PyTorch framework for training semantic segmentation models. This repo gives you a single interface to work with eight different segmentation architectures, handling the optimization and training specifics for each one.

## Supported Models

- **UNet** - Classic encoder-decoder architecture for biomedical image segmentation
- **FCN ResNet50/101** - Fully convolutional networks with ResNet backbones
- **DeepLabV3 ResNet50/101** - Atrous convolution-based models with ResNet backbones
- **DeepLabV3 MobileNet** - Lightweight DeepLabV3 with MobileNetV3 backbone
- **LRASPP MobileNet** - Lite R-ASPP for efficient mobile segmentation
- **SegNet** - Encoder-decoder with pooling indices for upsampling

## What's Included

- One training pipeline that works across all architectures
- Pre-trained weights support for transfer learning
- Logging split across different components (makes debugging way easier)
- Command-line config for everything
- Automatic checkpointing and evaluation
- Works with custom datasets at any resolution
- Built-in validation and testing
- Multi-GPU training support

## Setup

You'll need PyTorch and the usual suspects:

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

### All the Options

#### Model Setup
- `--module` - Pick your architecture (required)
- `--classes` - How many segmentation classes in your dataset (required)
- `--channels` - Image channels, usually 3 for RGB
- `--weights` - Load pre-trained weights (on by default)

#### Where Your Data Lives
- `--training-path` - Training data folder (required)
- `--validation-path` - Validation data folder (required)
- `--testing-path` - Test data folder (required)
- `--weights-path` - Existing checkpoint to continue from

#### Training Settings
- `--epochs` - Training epochs (default: 50)
- `--batch-size` - Batch size (default: 16)
- `--learning-rate` - Starting LR (default: 0.001)
- `--dimensions` - Image size as height width (default: 512 512)
- `--workers` - Data loading threads (default: 4)
- `--seed` - Set this for reproducible runs

#### If You Want to Get Fancy
- `--weight-decay` - L2 regularization (default: 0.0005)
- `--gamma` - LR decay rate (default: 0.1)
- `--parallelism` - Enable multi-GPU training (default: False)
- `--output` - Where to save trained model (default: model.pt)

### Real Example

```bash
python main.py \
  --module deeplabv3_resnet50 \
  --classes 19 \
  --training-path ./datasets/train \
  --validation-path ./datasets/val \
  --testing-path ./datasets/test \
  --dimensions 512 512 \
  --epochs 100 \
  --batch-size 8 \
  --learning-rate 0.001 \
  --weight-decay 0.0005 \
  --gamma 0.1 \
  --workers 8 \
  --seed 42 \
  --parallelism True \
  --output ./model.pt
```

## What Happens During Training

Pretty straightforward flow:

1. Model gets initialized with your settings
2. Data loaders spin up for training and validation
3. Training loop runs with automatic gradient updates
4. Validation happens after each epoch
5. Test evaluation runs at the end
6. Best weights get saved based on validation performance

## Logging

Logs are split into four files so you're not hunting through one massive log:

- `main.log` - Overall flow and status
- `loader.log` - Data loading stuff
- `modules.log` - Model operations
- `trainer.log` - Training metrics and performance

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

## Things Worth Knowing

Pre-trained weights load automatically when available. This usually helps a lot, especially if you don't have tons of training data.

Segmentation models are memory-intensive. If you run into OOM errors, try reducing the batch size or image dimensions.

Each model comes with sensible defaults, but you'll probably want to tweak things based on your dataset and how much compute you have available.

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.
