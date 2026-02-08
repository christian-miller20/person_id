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
python -m person_id_pi.cli identify data/clip.mp4 --store profiles/face_templates.json --verbose --auto-enroll-unknown
python -m person_id_pi.cli identify data/clip.mp4 --store profiles/face_templates.json --annotate-output runs/annotated.mp4
```

## How It Works (Minimal)

- **FaceEmbedder** uses InsightFace to detect faces and return embeddings.
- **IdentityEngine** aggregates embeddings within a clip into a stable tracklet embedding.
- **IdentityEngine** performs open-set matching (score + margin + evidence thresholds).
- **IdentityStore** persists per-user templates in `profiles/face_templates.json`.

## Notes

- Quality is now a composite: `det_score * size_score * blur_score` (see verbose logs).
- `identify` uses multi-face tracking by default (IoU-based tracker across frames).
- If `--annotate-output` is omitted, output defaults to `runs/<video_name>_annotated.mp4`.
- With `--verbose`, frame logs are written to `logs/<video_name>.log`; add `--tee-logs` to also mirror them to stdout.
- Use `--annotate-output path/to/out.mp4` to override annotated output location.
- The identity engine + template store are implemented and testable today.

## Output Metrics

`identify` prints one line per track:

- `track`: Temporary in-video track ID. Not a persistent identity.
- `accepted`: Whether the match passed open-set thresholds.
- `user_id`: Matched user if accepted, otherwise `None` (or auto-enrolled user when enabled).
- `score`: Best cosine similarity to any template for the best candidate user.
- `margin`: Difference between best and second-best user scores (higher means less ambiguity).
- `n_used`: Number of embeddings kept after quality filtering and outlier rejection.
- `dispersion`: Consistency of the tracklet embeddings (lower is more stable).
- `reason`: Decision reason, such as `accepted`, `below_threshold`, `insufficient_samples`, `high_dispersion`, or `auto_enrolled_unknown`.

Verbose frame logs (`--verbose`) include:

- `faces`: Number of detected faces in that frame.
- `quality`: Composite quality used for filtering (`det_score * size_score * blur_score`).
- `det`: Raw detector confidence. E.g. Do I see a face?
- `size`: Normalized face-size score. how well is face in frame?
- `blur`: Normalized sharpness score from Laplacian variance. Is face readable?

### Good Value Guidelines

These are practical heuristics for this repo's current defaults (not universal constants):

- `score`:
  - Good: `>= 0.70`
  - Borderline: `0.55-0.70`
  - Weak: `< 0.55`
- `margin`:
  - Good: `>= 0.12`
  - Borderline: `0.08-0.12`
  - Ambiguous: `< 0.08`
- `n_used`:
  - Minimum gate: `>= 5`
  - Better stability: `>= 15`
  - Strong evidence: `>= 30`
- `dispersion` (lower is better):
  - Very stable: `<= 0.08`
  - Usable: `0.08-0.20`
  - Risky/noisy: `> 0.20` (rejected if above configured max)
- Frame-level `quality`:
  - At/above filter: `>= 0.20`
  - Better matching runs: many frames in `0.20-0.35+`
  - Frequent `< 0.20`: expect fewer usable samples and more `unknown`
- Frame-level `det`/`size`/`blur`:
  - `det`: typically better when `>= 0.75`
  - `size`: near `1.0` is ideal (large face crop)
  - `blur`: higher is better; sustained low values usually drive poor identity evidence
