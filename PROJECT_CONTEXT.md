# Photo Enhancement Project Context

## Project Goal

This project aims to enhance old or low-quality photos through two core operations:

1. **Colorize** black-and-white photos using AI-based colorization models.
2. **Upscale** photos to higher resolutions with minimal quality loss using super-resolution models.

---

## Core Requirements

- Accept an input image (grayscale or low-resolution).
- Apply colorization and/or upscaling as requested.
- Output the enhanced image to a specified location.
- Each processing step should be independently usable.

---

## Colorization

- Convert grayscale (black-and-white) images to plausible color images.
- Use pretrained AI models (e.g., DeOldify, DDColor, or similar).
- Output should look natural and consistent with the original scene.

---

## Upscaling

- Increase image resolution (e.g., 2×, 4×) while preserving detail.
- Use pretrained super-resolution models (e.g., Real-ESRGAN, ESRGAN, SwinIR).
- Avoid introducing visible artifacts or blurring.

---

## Preferred Processing Flow

1. Load input image.
2. (Optional) Apply colorization if the image is grayscale.
3. (Optional) Apply upscaling to increase resolution.
4. Save the output image.

Steps 2 and 3 should be independently toggleable via command-line flags.

---

## Suggested Technical Direction

- **Language:** Python
- **Colorization model:** DeOldify or DDColor (pretrained, inference-only to start)
- **Super-resolution model:** Real-ESRGAN or SwinIR (pretrained, inference-only to start)
- **Image I/O:** Pillow or OpenCV
- **CLI:** `argparse` for a simple command-line interface

---

## Development Priorities

1. Build a working command-line prototype first — no UI.
2. Implement colorization module as a standalone component.
3. Implement upscaling module as a standalone component.
4. Integrate both into a single pipeline with optional flags.
5. Validate output quality on sample images.
6. Refactor and improve model choices or parameters as needed.

---

## Quality Goals

- Colorized images should look realistic and not over-saturated.
- Upscaled images should be sharp with no major artifacts.
- Processing time should be reasonable on a standard GPU or CPU.
- Code should be clean, modular, and easy to extend.

---

## Notes for AI Agents

- **Modularity is essential.** The colorization and upscaling components must be developed independently so each can be improved or swapped without affecting the other.
- **Start simple.** A working CLI prototype is the first milestone. Do not add a UI until the core pipeline is stable.
- **Use pretrained models.** Do not train models from scratch. Focus on inference pipelines using publicly available pretrained weights.
- **Avoid tight coupling.** Each module (colorization, upscaling) should expose a simple function interface (e.g., `colorize(image) -> image`, `upscale(image, scale) -> image`).
- **Test with the provided sample images** in the `input/` directory.

---

## Current Status

- **Phase:** Colorization module complete and tested.
- **Guidelines:** `.junie/guidelines.md` created for AI agent continuity.
- **Conda environment:** `photo_enhancement` (Python 3.10) — activate with `conda activate photo_enhancement`.
- **Dependencies:** `requirements.txt` created and installed (Pillow, OpenCV, NumPy, Real-ESRGAN, basicsr, gfpgan, requests, tqdm).
- **Sample images:** Two black-and-white photos colorized successfully and saved to `output/`.
- **README:** Written with setup instructions, usage examples, model weight download links, and project structure.
- **Modules implemented:**
  - `colorize.py` — OpenCV DNN colorization (Zhang et al., 2016); auto-downloads model weights; exposes `colorize(image) -> image`; CLI via `--input`/`--output`.
  - `download_models.py` — standalone helper to pre-download all model weights.
- **Model weights:** Stored in `models/` (auto-downloaded on first run via `colorize.py` or `download_models.py`).
  - `colorization_deploy_v2.prototxt` — from GitHub (richzhang/colorization)
  - `colorization_release_v2.caffemodel` — from Dropbox mirror (~129 MB)
  - `pts_in_hull.npy` — from GitHub (richzhang/colorization)
- **Modules implemented (continued):**
  - `upscale.py` — Real-ESRGAN super-resolution (2× and 4×); auto-downloads model weights; exposes `upscale(image, scale) -> image`; CLI via `--input`/`--output`/`--scale`; exits cleanly after batch processing via `sys.exit(0)`.
- **Compatibility fix:** Patched `basicsr/data/degradations.py` to import `rgb_to_grayscale` from `torchvision.transforms.functional` (removed in torchvision 0.17+).
- **Next step:** Implement the integrated pipeline (`enhance.py`) combining colorization and upscaling via CLI flags.

---

## Future Considerations

- Add a simple web UI or desktop GUI once the CLI pipeline is stable.
- Support batch processing of multiple images.
- Allow users to choose between multiple models per task.
- Add automatic detection of grayscale images to trigger colorization automatically.
- Explore fine-tuning models on domain-specific data (e.g., historical photos).
