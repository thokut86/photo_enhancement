"""Download colorization model weights from reliable mirrors."""
import urllib.request
from pathlib import Path

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

files = {
    "colorization_deploy_v2.prototxt": (
        "https://raw.githubusercontent.com/richzhang/colorization/"
        "caffe/models/colorization_deploy_v2.prototxt"
    ),
    "colorization_release_v2.caffemodel": (
        "https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1"
    ),
    "pts_in_hull.npy": (
        "https://github.com/richzhang/colorization/raw/"
        "caffe/resources/pts_in_hull.npy"
    ),
}

for name, url in files.items():
    dest = MODEL_DIR / name
    if dest.exists():
        print(f"Already exists: {name}")
        continue
    print(f"Downloading {name} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"  -> {dest} ({dest.stat().st_size:,} bytes)")

print("All done.")
