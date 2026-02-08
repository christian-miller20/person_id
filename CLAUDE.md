# Claude Notes

## Project Overview
Face-embedding + evidence-accumulation pipeline for cross-video identity. Single-person mode: select the highest-confidence face per frame.

## Key Commands

Install:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Enroll a user (creates templates):
```bash
python -m person_id_pi.cli enroll <user_id> data/<clip>.mp4 --store profiles/face_templates.json --verbose
```

Identify from a clip:
```bash
python -m person_id_pi.cli identify data/<clip>.mp4 --store profiles/face_templates.json --verbose
```

Identify + update templates on high-confidence matches:
```bash
python -m person_id_pi.cli identify data/<clip>.mp4 --store profiles/face_templates.json --verbose --update-templates
```

Run tests:
```bash
pytest -q
```

## Architecture (Minimal)

- `src/person_id_pi/face_embedder.py`: InsightFace detection + ArcFace embeddings.
- `src/person_id_pi/identity_engine.py`: aggregation + open-set matching + template update.
- `src/person_id_pi/identity_store.py`: JSON-backed template store.
- `src/person_id_pi/face_pipeline.py`: clip processing and verbose per-frame logging.
- `src/person_id_pi/cli.py`: `enroll`, `identify`, `list-users`, `delete-user`.

## Notes
- Quality in logs is InsightFace detector confidence (`det_score`).
- Aggregation: median anchor -> outlier reject -> mean of inliers.
- Matching uses max cosine similarity per user + margin to the second best.
