from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .face_embedder import FaceEmbedder
from .face_pipeline import ClipResult, FacePipeline
from .identity_config import IdentityConfig
from .identity_engine import IdentityEngine
from .identity_store import IdentityStore

app = typer.Typer(add_completion=False, help="Face-based identity pipeline.")


def _build_identity(store_path: Path) -> IdentityEngine:
    store = IdentityStore(store_path)
    config = IdentityConfig()
    return IdentityEngine(store=store, config=config)


@app.command()
def identify(
    source: str = typer.Argument(..., help="Path to video file or camera index."),
    store_path: Path = typer.Option(
        Path("profiles/face_templates.json"), "--store", help="Template store path."
    ),
    limit_frames: Optional[int] = typer.Option(
        None, "--limit-frames", help="Stop after N frames for quick checks."
    ),
    verbose: bool = typer.Option(
        True, "--verbose/--quiet", help="Print per-frame processing updates."
    ),
    update_templates: bool = typer.Option(
        False, "--update-templates/--no-update-templates", help="Update templates on high-confidence matches."
    ),
    multi_face: bool = typer.Option(
        False, "--multi-face/--single-face", help="Process and identify multiple faces in a clip."
    ),
    annotate_output: Optional[Path] = typer.Option(
        None,
        "--annotate-output",
        help="Write annotated output video with bounding boxes and identified user IDs.",
    ),
) -> None:
    identity = _build_identity(store_path)
    pipeline = FacePipeline(embedder=FaceEmbedder(), identity=identity)
    if multi_face:
        tracklets_by_id, frame_annotations = pipeline.extract_tracklets_with_annotations_from_video(
            source=source, limit_frames=limit_frames, verbose=verbose
        )
        decisions_by_track = {}
        for track_id in sorted(tracklets_by_id.keys()):
            tracklet = tracklets_by_id[track_id]
            decision = identity.match(tracklet)
            decisions_by_track[track_id] = decision
            if update_templates and decision.user_id and identity.should_update_templates(decision, tracklet):
                identity.update_templates(decision.user_id, tracklet)
            typer.secho(
                " ".join(
                    [
                        f"track={track_id}",
                        f"accepted={decision.accepted}",
                        f"user_id={decision.user_id}",
                        f"score={decision.score:.3f}",
                        f"margin={decision.margin:.3f}",
                        f"n_used={tracklet.n_used}",
                        f"dispersion={tracklet.dispersion:.3f}",
                        f"reason={decision.reason}",
                    ]
                ),
                fg=typer.colors.CYAN,
            )
        if annotate_output:
            pipeline.write_multi_face_annotations(
                source=source,
                output_path=annotate_output,
                frame_annotations=frame_annotations,
                decisions=decisions_by_track,
            )
            typer.secho(f"Annotated video written to {annotate_output}", fg=typer.colors.GREEN)
    else:
        tracklet = pipeline.extract_tracklet_from_video(
            source=source, limit_frames=limit_frames, verbose=verbose
        )
        decision = identity.match(tracklet)
        if update_templates and decision.user_id and identity.should_update_templates(decision, tracklet):
            identity.update_templates(decision.user_id, tracklet)
        result = ClipResult(
            decision_user_id=decision.user_id,
            decision_score=decision.score,
            decision_margin=decision.margin,
            accepted=decision.accepted,
            reason=decision.reason,
            n_used=tracklet.n_used,
            dispersion=tracklet.dispersion,
        )
        typer.secho(
            " ".join(
                [
                    f"accepted={result.accepted}",
                    f"user_id={result.decision_user_id}",
                    f"score={result.decision_score:.3f}",
                    f"margin={result.decision_margin:.3f}",
                    f"n_used={result.n_used}",
                    f"dispersion={result.dispersion:.3f}",
                    f"reason={result.reason}",
                ]
            ),
            fg=typer.colors.CYAN,
        )
        if annotate_output:
            label = decision.user_id if decision.accepted and decision.user_id else "unknown"
            pipeline.write_single_face_annotations(
                source=source,
                output_path=annotate_output,
                label=label,
                accepted=decision.accepted and decision.user_id is not None,
                limit_frames=limit_frames,
            )
            typer.secho(f"Annotated video written to {annotate_output}", fg=typer.colors.GREEN)


@app.command()
def enroll(
    user_id: str = typer.Argument(..., help="User ID to enroll."),
    source: str = typer.Argument(..., help="Path to video file or camera index."),
    store_path: Path = typer.Option(
        Path("profiles/face_templates.json"), "--store", help="Template store path."
    ),
    limit_frames: Optional[int] = typer.Option(
        None, "--limit-frames", help="Stop after N frames for quick checks."
    ),
    verbose: bool = typer.Option(
        True, "--verbose/--quiet", help="Print per-frame processing updates."
    ),
) -> None:
    identity = _build_identity(store_path)
    pipeline = FacePipeline(embedder=FaceEmbedder(), identity=identity)
    tracklet = pipeline.extract_tracklet_from_video(
        source=source, limit_frames=limit_frames, verbose=verbose
    )
    if tracklet.n_used < identity.config.n_min:
        typer.secho(
            f"Enrollment failed: only {tracklet.n_used} samples (min={identity.config.n_min})",
            fg=typer.colors.YELLOW,
        )
        return
    updated = identity.update_templates(user_id, tracklet)
    if updated:
        typer.secho(
            f"Enrolled {user_id}: n_used={tracklet.n_used} dispersion={tracklet.dispersion:.3f}",
            fg=typer.colors.GREEN,
        )
    else:
        typer.secho(
            f"Enrollment skipped: template too similar for {user_id}",
            fg=typer.colors.YELLOW,
        )


@app.command()
def list_users(
    store_path: Path = typer.Option(
        Path("profiles/face_templates.json"), "--store", help="Template store path."
    ),
) -> None:
    store = IdentityStore(store_path)
    for user_id in store.list_users():
        typer.echo(user_id)


@app.command()
def delete_user(
    user_id: str = typer.Argument(..., help="User ID to delete from the store."),
    store_path: Path = typer.Option(
        Path("profiles/face_templates.json"), "--store", help="Template store path."
    ),
) -> None:
    store = IdentityStore(store_path)
    if store.delete_user(user_id):
        typer.secho(f"Deleted {user_id}", fg=typer.colors.GREEN)
    else:
        typer.secho(f"User {user_id} not found", fg=typer.colors.YELLOW)


def run() -> None:
    app()


if __name__ == "__main__":
    run()
