# Person ID Pi

Face-embedding + evidence-accumulation pipeline for cross-video identity.

**System Architecture**
```
Video/Frames
   |
   v
[Face Detection + Alignment]
   |   (bbox, landmarks, det_score)
   v
[Embedding (ArcFace)]
   |   (512-d embedding, quality)
   v
[Tracklet Aggregation]
   |   (median -> outlier reject -> mean)
   v
[Identity Engine]
   |   (open-set match: score + margin + evidence)
   v
Identity Decision
   |
   v
[Template Store] <-> update on high-confidence matches
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## CLI

```bash
python -m person_id_pi.cli enroll alice data/clip.mp4 --store profiles/face_templates.json --verbose
python -m person_id_pi.cli identify data/clip.mp4 --store profiles/face_templates.json --verbose --update-templates
```

## How It Works (Minimal)

- **FaceEmbedder** uses InsightFace to detect faces and return embeddings.
- **IdentityEngine** aggregates embeddings within a clip into a stable tracklet embedding.
- **IdentityEngine** performs open-set matching (score + margin + evidence thresholds).
- **IdentityStore** persists per-user templates in `profiles/face_templates.json`.

## Notes

- Quality is now a composite: `det_score * size_score * blur_score` (see verbose logs).
- Multi-person tracking is not implemented yet; the pipeline chooses the highest-confidence face each frame.
- The identity engine + template store are implemented and testable today.
