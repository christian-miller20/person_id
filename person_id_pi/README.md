# person_id_pi

Face-embedding + evidence-accumulation pipeline for cross-video identity.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## CLI

```bash
python -m person_id_pi.cli add-user alice --store profiles/face_templates.json
python -m person_id_pi.cli identify data/clip.mp4 --store profiles/face_templates.json
```

Notes:
- The current `FaceEmbedder` is a stub. Wire a real detector + ArcFace model next.
- The identity engine + template store are implemented and testable today.
