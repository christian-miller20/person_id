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
python -m person_id_pi.cli identify-count data/clip.mp4 --store profiles/face_templates.json --events-store profiles/beverage_events.json --tee-logs
python -m person_id_pi.cli rename-user user_0002 bob --store profiles/face_templates.json
```

## How It Works (Minimal)

- **FaceEmbedder** uses InsightFace to detect faces and return embeddings.
- **IdentityEngine** aggregates embeddings within a clip into a stable tracklet embedding.
- **IdentityEngine** performs open-set matching (score + margin + evidence thresholds).
- **IdentityStore** persists per-user templates in `profiles/face_templates.json`.

## Notes

- Quality is now a composite: `det_score * size_score * blur_score` (see verbose logs).
- `identify` uses multi-face tracking by default (IoU-based tracker across frames).
- Tracks that age out (`track_max_age`) are finalized and included in end-of-run decisions.
- If `--annotate-output` is omitted, output defaults to `runs/<video_name>_annotated.mp4`.
- With `--verbose`, frame logs are written to `logs/<video_name>.log`; add `--tee-logs` to also mirror them to stdout.
- Use `--annotate-output path/to/out.mp4` to override annotated output location.
- The identity engine + template store are implemented and testable today.

## Identify + Count (Beverages)

`identify-count` runs face identity plus beverage detection/counting.

- Face boxes remain annotated with identity labels.
- Beverage objects are annotated in-video:
  - `BEER` for `cup` / `can` / `bottle`
  - `ESPRESSO` for `espresso_shot`
- Accepted face labels are rewritten with a running beer counter:
  - `user_id beers=<lifetime_before_run + beers_seen_in_this_video_so_far>`

Key flags:

- `--events-store profiles/beverage_events.json`: persisted beverage event store.
- `--count-beers/--no-count-beers`: enable/disable beer-like labels.
- `--count-espressos/--no-count-espressos`: enable/disable espresso label counting.
- `--beverage-hold-frames N`: keep beverage overlay labels visible for `N` frames after last detection (default `45`).
- `--object-conf-min`: override detector confidence threshold.
- `--association-max-dist`: override beverage-to-person association distance threshold.

Distinct-event controls in the pipeline:

- Minimum evidence gate: beverage track must be seen for multiple frames before counting.
- Cooldown dedupe: repeated events for the same `(user_id, label)` inside a cooldown window are rejected.

## Real-World Espresso Rollout

Use this process before enabling espresso counting in production clips.

1. Collect deployment-style data:
   - Capture from the real camera angle/location (kegerator viewpoint).
   - Include in-hand espresso shots, partial occlusions, and motion blur.
   - Include hard negatives: mugs, empty hands, cans/bottles, reflections.
2. Build reproducible train/val split:
   - Use `scripts/build_random_classify_split.py` for initial classification data split.
   - Keep a fixed random seed so you can compare experiments fairly.
3. Train espresso classifier baseline:
   - Start with `yolov8n-cls.pt`.
   - Use `imgsz=224` first; try `imgsz=320` if small-object detail is weak.
4. Validate on holdout:
   - Run `yolo classify val ...` and track `top1_acc`.
   - Ignore `top5_acc` for 2-class problems (`espresso` vs `non_espresso`).
5. Run real-clip smoke test:
   - Run prediction on unseen clips from your real environment.
   - Manually inspect false positives and false negatives.
6. Targeted data iteration:
   - Add 100-300 examples of the exact failure modes.
   - Retrain and compare against previous run using the same validation split.
7. Promotion criteria (practical):
   - Validation remains strong across retrains.
   - Real-clip false positives are low enough for your workflow.
   - In-hand espresso examples are consistently classified correctly.
8. Integrate into pipeline:
   - Keep beer detector path unchanged.
   - Add espresso as a secondary model path and monitor logs for a few runs.

## Future Work

- Implement kegerator activation triggers (door switch / motion / weight change) to capture short event clips instead of 24/7 recording.
- Implement dedicated espresso-shot recognition with a model/class set tuned for small cup/shot-glass objects.
- Add per-user beverage timelines and session reports (events per clip, per day, per week).
- Expose beverage dedupe/gating knobs directly in CLI for easier field tuning.

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
- `auto_enroll` (shown when `--auto-enroll-unknown` is enabled): per-track enrollment outcome, e.g. `enrolled` or `rejected:<reason>`.

Verbose frame logs (`--verbose`) include:

- `faces`: Number of detected faces in that frame.
- `quality`: Composite quality used for filtering (`det_score * size_score * blur_score`).
- `det`: Raw detector confidence. E.g. Do I see a face?
- `size`: Normalized face-size score. How large is the detected face in frame?
- `blur`: Normalized sharpness score from Laplacian variance. Is face readable?

`size` formula in this repo:

- `face_size_px = min(face_width_px, face_height_px)`
- `size = min(1.0, face_size_px / 160)`

Important: `size` reflects face pixel size, not centering/framing quality by itself.

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
