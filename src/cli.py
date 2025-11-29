from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .profile_tracker import ProfileTracker

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
        True,
        "--cups/--no-cups",
        help="Enable cup/wine glass/bottle detection and tagging.",
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
    )
    tracker.process_video(
        source=str(source),
        save_video_path=str(output) if output else None,
        limit_frames=limit_frames,
    )


def run() -> None:
    app()


if __name__ == "__main__":
    run()
