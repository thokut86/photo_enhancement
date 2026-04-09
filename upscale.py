"""
upscale.py — Upscaling module for the Photo Enhancement project.

Uses Real-ESRGAN for high-quality super-resolution upscaling.
Model weights are downloaded automatically on first run.

Usage:
    python upscale.py --input input/ --output output/
    python upscale.py --input input/photo.jpg --output output/photo_4x.jpg --scale 4
"""

import argparse
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


MODEL_DIR = Path("models")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# Real-ESRGAN model variants
MODELS = {
    2: {
        "name": "RealESRGAN_x2plus",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "filename": "RealESRGAN_x2plus.pth",
        "scale": 2,
    },
    4: {
        "name": "RealESRGAN_x4plus",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "filename": "RealESRGAN_x4plus.pth",
        "scale": 4,
    },
}


def download_file(url: str, dest: Path) -> None:
    """Download a file from url to dest with a progress indicator."""
    print(f"  Downloading {dest.name} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _progress(block_num, block_size, total_size):
        if total_size > 0:
            pct = min(block_num * block_size / total_size * 100, 100)
            print(f"\r  Progress: {pct:.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print(f"\r  Saved to {dest} ({dest.stat().st_size:,} bytes)")


def ensure_model(scale: int) -> Path:
    """Download the Real-ESRGAN model weights if not already present."""
    if scale not in MODELS:
        raise ValueError(f"Unsupported scale {scale}x. Choose from: {list(MODELS.keys())}")
    model_info = MODELS[scale]
    model_path = MODEL_DIR / model_info["filename"]
    if not model_path.exists():
        print(f"Model weights not found. Downloading {model_info['name']}...")
        download_file(model_info["url"], model_path)
    return model_path


def load_upsampler(scale: int):
    """Load and return a Real-ESRGAN upsampler for the given scale."""
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet

    model_path = ensure_model(scale)
    model_info = MODELS[scale]

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=model_info["scale"],
    )

    upsampler = RealESRGANer(
        scale=model_info["scale"],
        model_path=str(model_path),
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,
    )
    return upsampler


def upscale(image: Image.Image, scale: int = 4, upsampler=None) -> Image.Image:
    """
    Upscale a PIL image using Real-ESRGAN.

    Args:
        image:      Input PIL image.
        scale:      Upscaling factor (2 or 4).
        upsampler:  Pre-loaded RealESRGANer instance. Loaded automatically if None.

    Returns:
        Upscaled PIL image in RGB mode.
    """
    if upsampler is None:
        upsampler = load_upsampler(scale)

    # Convert PIL → OpenCV BGR (Real-ESRGAN expects BGR uint8)
    img_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)

    output_bgr, _ = upsampler.enhance(img_bgr, outscale=scale)

    # Convert BGR → RGB and back to PIL
    output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(output_rgb)


def process_path(input_path: Path, output_path: Path, scale: int, upsampler) -> None:
    """Upscale a single image file and save the result."""
    print(f"  Processing: {input_path.name}")
    with Image.open(input_path) as img:
        img.load()
    result = upscale(img, scale=scale, upsampler=upsampler)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_path = output_path.with_suffix(".jpg")
    result.save(save_path, quality=95)
    print(f"  Saved:      {save_path}  ({result.width}×{result.height})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upscale photos using Real-ESRGAN super-resolution."
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to an input image file or a directory of images."
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Path to an output image file or a directory for results."
    )
    parser.add_argument(
        "--scale", "-s", type=int, default=4, choices=[2, 4],
        help="Upscaling factor: 2 or 4 (default: 4)."
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"Loading Real-ESRGAN model ({args.scale}x)...")
    upsampler = load_upsampler(args.scale)
    print("Model loaded.\n")

    if input_path.is_dir():
        images = [
            f for f in input_path.iterdir()
            if f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if not images:
            print(f"No supported images found in {input_path}")
            return
        output_path.mkdir(parents=True, exist_ok=True)
        for img_file in sorted(images):
            out_file = output_path / img_file.name
            process_path(img_file, out_file, args.scale, upsampler)
    elif input_path.is_file():
        process_path(input_path, output_path, args.scale, upsampler)
    else:
        print(f"Error: '{input_path}' is not a valid file or directory.")
        raise SystemExit(1)

    print("\nDone.")
    sys.exit(0)


if __name__ == "__main__":
    main()
