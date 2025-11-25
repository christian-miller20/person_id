# YOLO Profile Tracker

This repository shows how to pair an Ultralytics YOLO detector with a light embedding store so a person spotted on camera can be recognized again in later frames. The pipeline uses YOLO for person detection and a pretrained CNN encoder (MobileNet V3 small by default) to build a cosine-similarity profile bank saved to `profiles/profiles.json`.

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download a YOLO checkpoint (e.g. `yolov8n.pt`) and place it in the project root or reference a custom path.

## Usage

Process a video while creating/updating person profiles:

```bash
python -m src.cli process data/example.mp4 --model yolov8n.pt --output runs/annotated.mp4
```

Useful options:
- `--device cuda:0` to use a GPU if available.
- `--encoder mobilenet_v3_small` keeps embeddings lightweight for edge devices; switch to `resnet18` if you need sharper identities.
- `--match-threshold 0.6` to demand higher similarity before reusing a profile.
- `--limit-frames 200` to stop early while iterating.

Profiles are stored as averaged embeddings and can be inspected via `profiles/profiles.json`. You can bootstrap the store with known subjects by capturing a few clean frames and letting the tracker run; subsequent sequences will reuse the saved IDs whenever cosine similarity stays above the threshold.

## Raspberry Pi 5 tips

- Follow the [Ultralytics Raspberry Pi guide](https://docs.ultralytics.com/guides/raspberry-pi/#install-ultralytics-package) when provisioning the board: enable a 4 GB swapfile, run `sudo apt update && sudo apt upgrade`, and install the system deps (`sudo apt install python3-pip python3-venv libopenblas-dev libjpeg-dev libfreetype6-dev`).
- Create an isolated environment and install Ultralytics plus this project inside it:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip wheel
  pip install ultralytics
  pip install -r requirements.txt
  ```
- Prefer the nano YOLO checkpoints (`yolov8n.pt`) and keep `--device cpu` unless you add a Coral TPU or similar accelerator.
- Use the default `mobilenet_v3_small` encoder; it reduces inference time by ~3× versus ResNet18 on ARM.
- Build OpenCV with SIMD (NEON) enabled (or install `opencv-python-headless`) and launch with `export OMP_NUM_THREADS=2` to stay within thermal limits.
- For real-time streams, set `--limit-frames` during debugging and lower `--conf` to ~0.3 so YOLO skips fewer detections per pass.
