# Junie Guidelines — Photo Enhancement Project

## Project Overview

This is a Python-based photo enhancement project. The goal is to colorize black-and-white photos and upscale photos to higher resolutions using pretrained AI models. See `PROJECT_CONTEXT.md` for full context.

---

## Key Rules for AI Agents

### Always Update PROJECT_CONTEXT.md
- After every meaningful development step, update `PROJECT_CONTEXT.md` to reflect the current state of the project.
- Document which modules exist, what has been implemented, what models are in use, and what the next steps are.
- This file is the single source of truth for any AI agent or developer picking up the project.

### Project Structure Conventions
- `input/` — source images to be processed (ignored by git)
- `output/` — processed/enhanced images (ignored by git)
- `colorize.py` or `colorization/` — colorization module
- `upscale.py` or `upscaling/` — upscaling module
- `enhance.py` — main CLI entry point combining both modules
- `models/` — pretrained model weights (should be gitignored if large)
- `requirements.txt` — Python dependencies

### Code Style
- Language: Python 3.10+
- Follow PEP 8.
- Each module (colorization, upscaling) must expose a simple function interface:
  - `colorize(image: PIL.Image) -> PIL.Image`
  - `upscale(image: PIL.Image, scale: int) -> PIL.Image`
- Use `argparse` for CLI interfaces.
- Use Pillow or OpenCV for image I/O.

### Modularity
- Colorization and upscaling must remain independent modules.
- Do not couple them — each should work standalone.
- The main pipeline (`enhance.py`) orchestrates them via flags.

### Models
- Use pretrained models only — do not train from scratch.
- Preferred colorization: OpenCV DNN-based colorization (lightweight, no heavy dependencies) or DDColor.
- Preferred super-resolution: Real-ESRGAN or SwinIR.
- Download model weights at runtime or document how to obtain them in `README.md`.

### Development Order
1. CLI prototype (colorization first)
2. Standalone upscaling module
3. Integrated pipeline
4. Quality validation
5. UI (only after CLI is stable)

---

## Workflow for Each Session

1. Read `PROJECT_CONTEXT.md` to understand current state and next steps.
2. Implement the next priority item.
3. Test with images in `input/`.
4. Update `PROJECT_CONTEXT.md` with what was done and what comes next.
5. Keep `requirements.txt` up to date.

---

## Notes

- The `input/` and `output/` directories are gitignored — do not commit images.
- Do not add a UI until the CLI pipeline is fully working.
- Keep dependencies minimal and well-documented.
