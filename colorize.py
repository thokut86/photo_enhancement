"""
colorize.py — Colorization module for the Photo Enhancement project.

Uses the OpenCV DNN-based colorization model (Zhang et al., 2016).
Model weights are downloaded automatically on first run.

Usage:
    python colorize.py --input input/ --output output/
    python colorize.py --input input/photo.jpg --output output/photo_color.jpg
"""

import argparse
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


MODEL_DIR = Path("models")
PROTOTXT_URL = (
    "https://raw.githubusercontent.com/richzhang/colorization/"
    "caffe/models/colorization_deploy_v2.prototxt"
)
CAFFEMODEL_URL = (
    "https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1"
)
PTS_URL = (
    "https://raw.githubusercontent.com/richzhang/colorization/"
    "caffe/resources/pts_in_hull.npy"
)

PROTOTXT_PATH = MODEL_DIR / "colorization_deploy_v2.prototxt"
CAFFEMODEL_PATH = MODEL_DIR / "colorization_release_v2.caffemodel"
PTS_PATH = MODEL_DIR / "pts_in_hull.npy"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def download_file(url: str, dest: Path) -> None:
    """Download a file from url to dest with a progress indicator."""
    print(f"  Downloading {dest.name} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved to {dest}")


def ensure_models() -> None:
    """Download model weights if they are not already present."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not PROTOTXT_PATH.exists():
        download_file(PROTOTXT_URL, PROTOTXT_PATH)
    if not CAFFEMODEL_PATH.exists():
        download_file(CAFFEMODEL_URL, CAFFEMODEL_PATH)
    if not PTS_PATH.exists():
        download_file(PTS_URL, PTS_PATH)


def load_model() -> cv2.dnn_Net:
    """Load the colorization network and inject cluster centers."""
    ensure_models()
    net = cv2.dnn.readNetFromCaffe(str(PROTOTXT_PATH), str(CAFFEMODEL_PATH))
    pts = np.load(str(PTS_PATH))
    # Add the cluster centers as 1×1 convolution weights
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype(np.float32)]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype=np.float32)]
    return net


def colorize(image: Image.Image, net: cv2.dnn_Net | None = None) -> Image.Image:
    """
    Colorize a grayscale (or color) PIL image.

    Args:
        image: Input PIL image.
        net:   Pre-loaded cv2 DNN network. Loaded automatically if None.

    Returns:
        Colorized PIL image in RGB mode.
    """
    if net is None:
        net = load_model()

    # Convert PIL → OpenCV BGR
    img_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)

    # Work in LAB color space; use only the L channel as input
    img_lab = cv2.cvtColor(img_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    l_channel = img_lab[:, :, 0]

    # Resize L channel to the network's expected 224×224 input
    l_resized = cv2.resize(l_channel, (224, 224))
    l_resized -= 50  # mean-center

    blob = cv2.dnn.blobFromImage(l_resized)
    net.setInput(blob)
    ab_pred = net.forward()[0, :, :, :].transpose((1, 2, 0))

    # Resize predicted AB channels back to original size
    h, w = img_bgr.shape[:2]
    ab_resized = cv2.resize(ab_pred, (w, h))

    # Combine original L with predicted AB
    lab_out = np.concatenate([l_channel[:, :, np.newaxis], ab_resized], axis=2)
    bgr_out = np.clip(cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR), 0, 1)
    rgb_out = (bgr_out[:, :, ::-1] * 255).astype(np.uint8)

    return Image.fromarray(rgb_out)


def process_path(input_path: Path, output_path: Path, net: cv2.dnn_Net) -> None:
    """Colorize a single image file and save the result."""
    print(f"  Processing: {input_path.name}")
    img = Image.open(input_path)
    result = colorize(img, net)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Always save as high-quality JPEG
    save_path = output_path.with_stem(output_path.stem + "_colorized").with_suffix(".jpg")
    result.save(save_path, quality=95)
    print(f"  Saved:      {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Colorize black-and-white photos using a pretrained DNN model."
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to an input image file or a directory of images."
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Path to an output image file or a directory for results."
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print("Loading colorization model...")
    net = load_model()
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
            out_file = output_path / img_file.stem
            process_path(img_file, out_file, net)
    elif input_path.is_file():
        process_path(input_path, output_path, net)
    else:
        print(f"Error: '{input_path}' is not a valid file or directory.")
        raise SystemExit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
