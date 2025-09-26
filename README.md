# Semantic Segmentation Training Framework

A PyTorch framework for training segmentation models on datasets with XML polygon annotations. Supports eight different architectures and handles the conversion from polygon coordinates to pixel-level masks automatically.

## What's Included

- **Eight model architectures** from basic UNet to modern DeepLabV3 variants
- **XML polygon support** that converts coordinate lists to segmentation masks
- **Progress tracking** with async bars that show training metrics in real-time  
- **Multi-GPU support** for faster training on multi-card systems
- **Detailed logging** split across different components for easier debugging
- **Pre-trained initialization** available for most torchvision-based models

## Available Models

| Model Name | Backbone | Pre-trained | Speed | Accuracy | Memory |
|------------|----------|-------------|-------|----------|---------|
| `unet` | Custom encoder-decoder | No | Medium | High | Medium |
| `segnet` | VGG-style with pooling indices | No | Medium | High | Low |
| `fcn_resnet50` | ResNet50 | Yes | Fast | Medium | Medium |
| `fcn_resnet101` | ResNet101 | Yes | Medium | High | High |
| `deeplabv3_resnet50` | ResNet50 + ASPP | Yes | Medium | High | Medium |
| `deeplabv3_resnet101` | ResNet101 + ASPP | Yes | Slow | Very High | High |
| `deeplabv3_mobilenet` | MobileNetV3 + ASPP | Yes | Fast | Medium | Low |
| `lraspp_mobilenet` | MobileNetV3 + Lite ASPP | Yes | Very Fast | Medium | Very Low |

## Dataset Setup

Your data needs to be organized like this:

```
dataset/
├── images/          # Your actual images
│   ├── image001.jpg
│   ├── image002.png
│   └── ...
└── annotations/     # XML files with same base names
    ├── image001.xml
    ├── image002.xml
    └── ...
```

The XML files should have this structure:
```xml
<annotation>
    <object>
        <name>class</name>  <!-- or whatever your class is called -->
        <polygon>
            <x1>150.2</x1>
            <y1>200.5</y1>
            <x2>180.7</x2>
            <y2>195.3</y2>
            <!-- keep adding x,y pairs to define the polygon -->
        </polygon>
    </object>
    <!-- you can have multiple objects per image -->
</annotation>
```

## Getting Started

Install the requirements first:
```bash
pip install torch torchvision numpy pillow
```

Then run a basic training session:
```bash
python main.py \
    --module unet \
    --classes 3 \
    --training-path ./dataset/train \
    --validation-path ./dataset/val \
    --testing-path ./dataset/test \
    --epochs 50
```

## Command Line Options

### Required Parameters
- `-m, --module` - Which model to use (pick from the table above)
- `-c, --classes` - How many classes you're segmenting
- `-tp, --training-path` - Where your training data lives
- `-vp, --validation-path` - Where your validation data lives  
- `-tep, --testing-path` - Where your test data lives

### Optional Parameters
- `-w, --weights` - Use pre-trained weights (default: True)
- `-ch, --channels` - Input channels, 3 for RGB or 1 for grayscale (default: 3)
- `-wp, --weights-path` - Load existing model weights from this path
- `-d, --dimensions` - Image size as height width (default: 512 512)
- `-e, --epochs` - How many epochs to train (default: 50)
- `-b, --batch-size` - Batch size, bigger needs more memory (default: 16)
- `-lr, --learning-rate` - Learning rate for the optimizer (default: 0.001)
- `-wk, --workers` - Number of data loading workers (default: 4)
- `-s, --seed` - Random seed for reproducible results
- `-wd, --weight-decay` - Weight decay to prevent overfitting (default: 0.0005)
- `-g, --gamma` - Learning rate decay factor (default: 0.1)
- `-p, --parallelism` - Use multiple GPUs if available (default: False)
- `-o, --output` - Where to save the final model (default: model.pt)

## Model Architecture Notes

**UNet** works really well for medical images and anywhere you need precise edges. The skip connections help it keep track of fine details while still understanding the big picture.

**SegNet** is memory-efficient because it uses pooling indices instead of storing skip connections. Good choice when you're running low on GPU memory.

**FCN models** are solid general-purpose options. ResNet50 is faster, ResNet101 is more accurate but slower.

**DeepLabV3** excels when your images have objects at different scales. The ASPP module looks at multiple scales simultaneously, which helps a lot with complex scenes.

**Mobile variants** trade some accuracy for speed. LRASPP is the fastest option here - use it for real-time applications.

## Output Files

When you run training, you'll get several log files:
- `main.log` - Overall program status and any major errors
- `loader.log` - Dataset loading issues and statistics  
- `modules.log` - Model setup and architecture details
- `trainer.log` - Training progress, loss curves, and metrics

Your final model gets saved to whatever you specify with `--output` (defaults to `model.pt`).

## Performance Tips

**Running out of memory?**
- Lower your batch size with `--batch-size 8` or even `--batch-size 4`
- Use smaller images with `--dimensions 256 256`
- Switch to a mobile model like `lraspp_mobilenet`

**Training too slow?**
- Increase data loading workers with `--workers 8` 
- Use multi-GPU training with `--parallelism True`
- Try a faster model like `fcn_resnet50` instead of the ResNet101 variants

**No matching files found?**
- Double-check that your image and XML files have the same base names
- Make sure you're using supported image formats (jpg, png, jpeg, tiff)
- Verify your XML files actually contain `<object>` and `<polygon>` tags

## Common Problems

The most frequent issue is memory errors during training. Start with a small batch size and work your way up. If you're still having problems, try reducing image dimensions or switching to a lighter model.

Dataset loading problems usually come from mismatched filenames or incorrect XML structure. Check the log files - `loader.log` will tell you exactly what went wrong.

Model loading failures typically happen when you misspell the architecture name or try to load weights that don't match your current model setup.

## Architecture Recommendations

- **Medical imaging**: UNet or SegNet
- **Natural images**: DeepLabV3 ResNet50 or FCN ResNet101  
- **Real-time applications**: LRASPP MobileNet
- **High accuracy needed**: DeepLabV3 ResNet101
- **Limited GPU memory**: SegNet or Mobile variants
- **First time experimenting**: FCN ResNet50

The framework handles most of the complexity for you - polygon conversion, data loading, progress tracking, and model saving all happen automatically.
