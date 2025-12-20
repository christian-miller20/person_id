from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer

from .profile_store import ProfileStore
from .profile_tracker import ProfileTracker

app = typer.Typer(add_completion=False, help="Profile-aware YOLO pipeline.")


@app.command()
def consume(
    source: str = typer.Argument(..., help="Path to video file or camera index."),
    model: str = typer.Option("yolov8n.pt", "--model", "-m", help="YOLO model path."),
    profiles_path: Path = typer.Option(
        Path("profiles/profiles.json"), "--profiles", help="Profile store path."
    ),
    device: str = typer.Option("cpu", "--device", help="Inference device string."),
    encoder: str = typer.Option(
        "mobilenet_v3_small",
        "--encoder",
        help="Embedding backbone (mobilenet_v3_small|mobilenet_v3_large|resnet18|reid_torchscript).",
    ),
    reid_model: Optional[Path] = typer.Option(
        None,
        "--reid-model",
        help="TorchScript person ReID model path (required for encoder=reid_torchscript).",
    ),
    conf: float = typer.Option(0.5, "--conf", help="Confidence threshold for YOLO."),
    match_threshold: float = typer.Option(
        0.55, "--match-threshold", help="Cosine similarity needed to reidentify."
    ),
    profile_window: int = typer.Option(
        5,
        "--profile-window",
        help="Number of recent embeddings to average per profile (0 disables window).",
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Optional path to save annotated video."
    ),
    limit_frames: Optional[int] = typer.Option(
        None, "--limit-frames", help="Stop after N frames for quick checks."
    ),
    enable_drinks: bool = typer.Option(
        True,
        "--drinks/--no-drinks",
        help="Detect bottle/cup/wine glass and treat it as a beer proxy.",
    ),
    temporal: bool = typer.Option(
        True,
        "--temporal/--no-temporal",
        help="Use IoU-based temporal association to stabilize profile IDs.",
    ),
    track_iou_threshold: float = typer.Option(
        0.3,
        "--track-iou-threshold",
        help="IoU needed to associate detections to the previous frame track.",
    ),
    track_max_age: int = typer.Option(
        10,
        "--track-max-age",
        help="Drop a track after N missed frames (0 disables tracking state).",
    ),
    keep_threshold: Optional[float] = typer.Option(
        None,
        "--keep-threshold",
        help="Cosine similarity needed to keep a track's current profile ID (defaults to match_threshold - 0.1).",
    ),
    warmup_threshold: Optional[float] = typer.Option(
        None,
        "--warmup-threshold",
        help="Temporary cosine threshold to use during warmup frames (disabled by default).",
    ),
    warmup_frames: int = typer.Option(
        0,
        "--warmup-frames",
        help="Number of initial frames to apply the warmup threshold.",
    ),
    ema_alpha: Optional[float] = typer.Option(
        0.2,
        "--ema-alpha",
        help="Smoothing factor for the long-term EMA embedding (use 0 to disable).",
    ),
    min_primary_frames: int = typer.Option(
        3,
        "--min-primary-frames",
        help="Wait for a profile to dominate this many frames before counting a beer.",
    ),
    min_beer_frames: int = typer.Option(
        1,
        "--min-beer-frames",
        help="Require this many frames with a drink proxy before counting a beer.",
    ),
) -> None:
    """Process a clip and increment beers consumed for the dominant profile once."""
    if output is None:
        input_name = Path(source).name or "capture"
        output = Path("runs") / f"annotated_consume_{input_name}"
    tracker = ProfileTracker(
        model_name=model,
        profiles_path=profiles_path,
        device=device,
        encoder=encoder,
        reid_model_path=reid_model,
        conf=conf,
        match_threshold=match_threshold,
        profile_window=profile_window,
        enable_cup_detection=enable_drinks,
        enable_temporal_tracking=temporal,
        track_iou_threshold=track_iou_threshold,
        track_max_age=track_max_age,
        keep_threshold=keep_threshold,
        warmup_threshold=warmup_threshold,
        warmup_frames=warmup_frames,
        ema_alpha=ema_alpha,
    )
    tracker.process_video(
        source=str(source),
        save_video_path=str(output) if output else None,
        limit_frames=limit_frames,
        count_beers=True,
        min_primary_frames=min_primary_frames,
        min_beer_frames=min_beer_frames,
    )


@app.command()
def delete(
    profile_ids: List[str] = typer.Argument(..., help="One or more profile IDs to remove."),
    profiles_path: Path = typer.Option(
        Path("profiles/profiles.json"), "--profiles", help="Profile store path."
    ),
) -> None:
    """Delete one or more profiles from the store."""
    store = ProfileStore(profiles_path)
    for profile_id in profile_ids:
        if store.delete_profile(profile_id):
            typer.secho(f"Deleted {profile_id}", fg=typer.colors.GREEN)
        else:
            typer.secho(f"Profile {profile_id} not found", fg=typer.colors.YELLOW)


def run() -> None:
    app()


if __name__ == "__main__":
    run()
