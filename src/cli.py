from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer

from .profile_tracker import ProfileTracker
from .profile_store import ProfileStore

app = typer.Typer(add_completion=False, help="Profile-aware YOLO pipeline.")


@app.command()
def process(
    source: str = typer.Argument(..., help="Path to video file or camera index."),
    model: str = typer.Option("yolov8n.pt", "--model", "-m", help="YOLO model path."),
    profiles_path: Path = typer.Option(
        Path("profiles/profiles.json"), "--profiles", help="Profile store path."
    ),
    device: str = typer.Option("cpu", "--device", help="Inference device string."),
    encoder: str = typer.Option(
        "mobilenet_v3_small",
        "--encoder",
        help="Embedding backbone (mobilenet_v3_small|mobilenet_v3_large|resnet18).",
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
    enable_cups: bool = typer.Option(
        False,
        "--cups/--no-cups",
        help="Enable cup/wine glass/bottle detection and tagging.",
    ),
    force_profile: Optional[str] = typer.Option(
        None,
        "--force-profile",
        help="Assign every detection to this profile ID (overrides matching).",
    ),
    warmup_threshold: Optional[float] = typer.Option(
        None,
        "--warmup-threshold",
        help="Temporary cosine threshold to use during warmup frames.",
    ),
    warmup_frames: int = typer.Option(
        0,
        "--warmup-frames",
        help="Number of initial frames to apply the warmup threshold.",
    ),
) -> None:
    """Run the YOLO + embedding pipeline and update the profile store."""
    tracker = ProfileTracker(
        model_name=model,
        profiles_path=profiles_path,
        device=device,
        encoder=encoder,
        conf=conf,
        match_threshold=match_threshold,
        profile_window=profile_window,
        enable_cup_detection=enable_cups,
        force_profile_id=force_profile,
        warmup_threshold=warmup_threshold,
        warmup_frames=warmup_frames,
    )
    tracker.process_video(
        source=str(source),
        save_video_path=str(output) if output else None,
        limit_frames=limit_frames,
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
