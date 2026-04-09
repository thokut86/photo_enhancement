# Photo Enhancement

A Python-based CLI tool to enhance old or low-quality photos using pretrained AI models. It supports two core operations:

1. **Colorization** — convert black-and-white photos to realistic color images.
2. **Upscaling** — increase photo resolution with minimal quality loss.

---

## Features

- AI-powered colorization using OpenCV DNN-based pretrained models
- Super-resolution upscaling via Real-ESRGAN
- Modular design — colorization and upscaling work independently
- Simple command-line interface
- Batch-friendly: process images from an `input/` folder to an `output/` folder

---

## Project Structure

```
photo_enhancement/
├── input/              # Place source images here (gitignored)
├── output/             # Enhanced images are saved here (gitignored)
├── models/             # Pretrained model weights (gitignored if large)
├── colorize.py         # Colorization module
├── upscale.py          # Upscaling module
├── enhance.py          # Main CLI entry point
├── requirements.txt    # Python dependencies
└── PROJECT_CONTEXT.md  # Developer/agent context and status
```

---

## Requirements

- Python 3.10+
- [Conda](https://docs.conda.io/) (recommended for environment management)

---

## Setup

### 1. Create and activate the conda environment

```bash
conda create -n photo_enhancement python=3.10 -y
conda activate photo_enhancement
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Model Weights

### Colorization (OpenCV DNN)

Download the following files and place them in the `models/` directory:

| File | URL |
|------|-----|
| `colorization_deploy_v2.prototxt` | [Download](https://github.com/richzhang/colorization/blob/caffe/models/colorization_deploy_v2.prototxt) |
| `colorization_release_v2.caffemodel` | [Download](http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel) |
| `pts_in_hull.npy` | [Download](https://github.com/richzhang/colorization/blob/caffe/resources/pts_in_hull.npy) |

### Upscaling (Real-ESRGAN)

Model weights are downloaded automatically at runtime.

---

## Usage

### Colorize images

```bash
python colorize.py --input input/ --output output/
```

### Upscale images

```bash
python upscale.py --input input/ --output output/ --scale 4
```

### Full pipeline (colorize + upscale)

```bash
python enhance.py --input input/ --output output/ --colorize --upscale --scale 4
```

---

## Development Status

See [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) for the full development context, current status, and next steps.

---

## Quality Goals

- Colorized images look realistic and natural, not over-saturated.
- Upscaled images are sharp with no major artifacts.
- Processing is reasonable on both CPU and GPU.

---

## Future Plans

- Web UI or desktop GUI (after CLI is stable)
- Automatic grayscale detection
- Support for multiple model backends
- Batch processing improvements
