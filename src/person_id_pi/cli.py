from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .face_embedder import FaceEmbedder
from .face_pipeline import FacePipeline
from .face_types import IdentityDecision
from .identity_config import IdentityConfig
from .identity_engine import IdentityEngine
from .identity_store import IdentityStore

app = typer.Typer(add_completion=False, help="Face-based identity pipeline.")


def _build_identity(store_path: Path) -> IdentityEngine:
    store = IdentityStore(store_path)
    config = IdentityConfig()
    return IdentityEngine(store=store, config=config)


def _default_annotate_output_path(source: str) -> Path:
    src = Path(source)
    stem = src.stem if src.suffix else src.name
    return Path("runs") / f"{stem}_annotated.mp4"


def _default_log_path(source: str) -> Path:
    src = Path(source)
    stem = src.stem if src.suffix else src.name
    return Path("logs") / f"{stem}.log"


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
        False,
        "--update-templates/--no-update-templates",
        help="Update templates on high-confidence matches.",
    ),
    annotate_output: Optional[Path] = typer.Option(
        None,
        "--annotate-output",
        help="Write annotated output video with bounding boxes and identified user IDs.",
    ),
    auto_enroll_unknown: bool = typer.Option(
        False,
        "--auto-enroll-unknown/--no-auto-enroll-unknown",
        help="Automatically enroll unknown identities with generated user IDs.",
    ),
    tee_logs: bool = typer.Option(
        False,
        "--tee-logs/--no-tee-logs",
        help="Also mirror verbose frame logs to stdout while writing logs/<video_input>.log.",
    ),
) -> None:
    identity = _build_identity(store_path)
    pipeline = FacePipeline(embedder=FaceEmbedder(), identity=identity)
    output_path = annotate_output or _default_annotate_output_path(source)
    log_path = _default_log_path(source)
    log_fn = None
    if verbose:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("w", encoding="utf-8")
        typer.secho(f"Verbose logs written to {log_path}", fg=typer.colors.BLUE)
        log_fn = (
            (lambda message: (log_handle.write(f"{message}\n"), print(message))[0])
            if tee_logs
            else (lambda message: log_handle.write(f"{message}\n"))
        )

    try:
        tracklets_by_id, frame_annotations = (
            pipeline.extract_tracklets_with_annotations_from_video(
                source=source,
                limit_frames=limit_frames,
                verbose=verbose,
                log_fn=log_fn,
            )
        )
        decisions_by_track = {}
        for track_id in sorted(tracklets_by_id.keys()):
            tracklet = tracklets_by_id[track_id]
            decision = identity.match(tracklet)
            auto_enroll_status: Optional[str] = None
            # if match, update embedding for user if update_templates is enabled
            if (
                update_templates
                and decision.user_id
                and identity.should_update_templates(decision, tracklet)
            ):
                identity.update_templates(decision.user_id, tracklet)
            # if not a match and auto_enroll_unknown is enabled, create new user and enroll
            if auto_enroll_unknown:
                block_reason = identity.auto_enroll_block_reason(decision, tracklet)
                if block_reason is None:
                    new_user_id = identity.store.generate_new_user_id()
                    added = identity.update_templates(new_user_id, tracklet)
                    if added:
                        decision = IdentityDecision(
                            user_id=new_user_id,
                            score=decision.score,
                            margin=decision.margin,
                            accepted=True,
                            reason="auto_enrolled_unknown",
                        )
                        auto_enroll_status = "enrolled"
                    else:
                        auto_enroll_status = "rejected:template_too_similar"
                else:
                    auto_enroll_status = f"rejected:{block_reason}"
            decisions_by_track[track_id] = decision
            fields = [
                f"track={track_id}",
                f"accepted={decision.accepted}",
                f"user_id={decision.user_id}",
                f"score={decision.score:.3f}",
                f"margin={decision.margin:.3f}",
                f"n_used={tracklet.n_used}",
                f"dispersion={tracklet.dispersion:.3f}",
                f"reason={decision.reason}",
            ]
            if auto_enroll_unknown:
                fields.append(f"auto_enroll={auto_enroll_status}")
            typer.secho(" ".join(fields), fg=typer.colors.CYAN)
        pipeline.write_multi_face_annotations(
            source=source,
            output_path=output_path,
            frame_annotations=frame_annotations,
            decisions=decisions_by_track,
        )
        typer.secho(f"Annotated video written to {output_path}", fg=typer.colors.GREEN)
    finally:
        if verbose:
            log_handle.close()

# @app.command()
# def identify_count(    
#         source: str = typer.Argument(..., help="Path to video file or camera index."),
#         store_path: Path = typer.Option(
#             Path("profiles/face_templates.json"), "--store", help="Template store path."
#         ),
#         limit_frames: Optional[int] = typer.Option(
#             None, "--limit-frames", help="Stop after N frames for quick checks."
#         ),
#         verbose: bool = typer.Option(
#             True, "--verbose/--quiet", help="Print per-frame processing updates."
#         ),
#         update_templates: bool = typer.Option(
#             False,
#             "--update-templates/--no-update-templates",
#             help="Update templates on high-confidence matches.",
#         ),
#         annotate_output: Optional[Path] = typer.Option(
#             None,
#             "--annotate-output",
#             help="Write annotated output video with bounding boxes and identified user IDs.",
#         ),
#         auto_enroll_unknown: bool = typer.Option(
#             False,
#             "--auto-enroll-unknown/--no-auto-enroll-unknown",
#             help="Automatically enroll unknown identities with generated user IDs.",
#         ),
#         tee_logs: bool = typer.Option(
#             False,
#             "--tee-logs/--no-tee-logs",
#             help="Also mirror verbose frame logs to stdout while writing logs/<video_input>.log.",
#         ),
#         events_store: Path = typer.Option(
#             Path("profiles/beverage_events.json"),
#             "--events-store",
#             help="Path to store detected beverage events.",
#         ),
#         count_beers: bool = typer.Option(
#             True,
#             "--count-beers/--no-count-beers",
#             help="Whether to count beer consumption events.",
#         ),
#         count_espressos: bool = typer.Option(
#             False,
#             "--count-espressos/--no-count-espressos",
#             help="Whether to count espresso consumption events.",
#         ),
#         beverage_annotate: bool = typer.Option(
#             True,
#             "--beverage-annotate/--no-beverage-annotate",
#             help="Whether to annotate detected beverage events in the output video",
#         ),
#         detector_model: Optional[str] = typer.Option(
#             None,
#             "--detector-model",
#             help="Optional custom detector model path or name for beverage event detection.",
#         ),
#         object_conf_min: Optional[float] = typer.Option(
#             None,
#             "--object-conf-min",
#             help="Optional override for minimum object confidence threshold for beverage event detection.",
#         ),
#         association_max_dist: Optional[float] = typer.Option(
#             None,
#             "--association-max-dist",
#             help="Optional override for maximum distance threshold for associating beverage events with identified users.",
#         ),
#     ) -> None:


    

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
    tracklet = pipeline.extract_primary_tracklet_from_video(
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


@app.command()
def rename_user(
    cur_user_id: str = typer.Argument(..., help="Current user ID to rename."),
    new_user_id: str = typer.Argument(..., help="New user ID."),
    store_path: Path = typer.Option(
        Path("profiles/face_templates.json"), "--store", help="Template store path."
    ),
) -> None:
    store = IdentityStore(store_path)
    if not store.has_user(cur_user_id):
        typer.secho(f"User {cur_user_id} not found", fg=typer.colors.YELLOW)
        return
    if cur_user_id != new_user_id and store.has_user(new_user_id):
        typer.secho(f"User {new_user_id} already exists", fg=typer.colors.YELLOW)
        return
    renamed = store.rename_user(cur_user_id, new_user_id)
    if not renamed:
        typer.secho(
            f"Unable to rename {cur_user_id} to {new_user_id}", fg=typer.colors.YELLOW
        )
        return
    typer.secho(f"Renamed {cur_user_id} to {new_user_id}", fg=typer.colors.GREEN)


def run() -> None:
    app()


if __name__ == "__main__":
    run()
