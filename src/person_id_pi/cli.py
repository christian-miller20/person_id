from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .cli_helpers import (
    build_log_writer,
    build_running_beer_label_resolver,
    default_annotate_output_path,
    run_beverage_stage,
    run_face_stage,
)
from .face_embedder import FaceEmbedder
from .face_pipeline import FacePipeline
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
    output_path = annotate_output or default_annotate_output_path(source)
    log_fn, log_handle = build_log_writer(
        source=source,
        verbose=verbose,
        tee_logs=tee_logs,
    )
    try:
        run_face_stage(
            source=source,
            identity=identity,
            pipeline=pipeline,
            output_path=output_path,
            limit_frames=limit_frames,
            verbose=verbose,
            log_fn=log_fn,
            update_templates=update_templates,
            auto_enroll_unknown=auto_enroll_unknown,
        )
    finally:
        if log_handle is not None:
            log_handle.close()


@app.command("identify-count")
def identify_count(
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
    events_store: Path = typer.Option(
        Path("profiles/beverage_events.json"),
        "--events-store",
        help="Path to store detected beverage events.",
    ),
    count_beers: bool = typer.Option(
        True,
        "--count-beers/--no-count-beers",
        help="Whether to count beer consumption events.",
    ),
    count_espressos: bool = typer.Option(
        False,
        "--count-espressos/--no-count-espressos",
        help="Whether to count espresso consumption events.",
    ),
    detector_model: Optional[str] = typer.Option(
        None,
        "--detector-model",
        help="Optional custom detector model path or name for beverage event detection.",
    ),
    object_conf_min: Optional[float] = typer.Option(
        None,
        "--object-conf-min",
        help="Optional override for minimum object confidence threshold for beverage event detection.",
    ),
    association_max_dist: Optional[float] = typer.Option(
        None,
        "--association-max-dist",
        help="Optional override for maximum distance threshold for associating beverage events with identified users.",
    ),
    beverage_hold_frames: int = typer.Option(
        45,
        "--beverage-hold-frames",
        help="How many frames to keep BEER/ESPRESSO overlay visible after last detection.",
    ),
) -> None:
    identity = _build_identity(store_path)
    pipeline = FacePipeline(embedder=FaceEmbedder(), identity=identity)
    output_path = annotate_output or default_annotate_output_path(source)
    log_fn, log_handle = build_log_writer(
        source=source,
        verbose=verbose,
        tee_logs=tee_logs,
    )

    try:
        face_result = run_face_stage(
            source=source,
            identity=identity,
            pipeline=pipeline,
            output_path=output_path,
            limit_frames=limit_frames,
            verbose=verbose,
            log_fn=log_fn,
            update_templates=update_templates,
            auto_enroll_unknown=auto_enroll_unknown,
        )
        beverage_result = run_beverage_stage(
            source=source,
            annotate_output_path=output_path,
            face_result=face_result,
            events_store=events_store,
            count_beers=count_beers,
            count_espressos=count_espressos,
            detector_model=detector_model,
            object_conf_min=object_conf_min,
            association_max_dist=association_max_dist,
            limit_frames=limit_frames,
            beverage_hold_frames=beverage_hold_frames,
            verbose=verbose,
            log_fn=log_fn,
        )
        this_run_persisted_beers: dict[str, int] = {}
        for event in beverage_result.persisted_events:
            if event.beverage_label not in {"cup", "can", "bottle"}:
                continue
            this_run_persisted_beers[event.user_id] = (
                this_run_persisted_beers.get(event.user_id, 0) + 1
            )
        baseline_beers_by_user = {
            user_id: max(0, total - this_run_persisted_beers.get(user_id, 0))
            for user_id, total in beverage_result.beer_totals.items()
        }
        running_beer_label = build_running_beer_label_resolver(
            beverage_result.events,
            baseline_beers_by_user=baseline_beers_by_user,
        )
        pipeline.write_multi_face_annotations_in_place(
            video_path=output_path,
            frame_annotations=face_result.frame_annotations,
            decisions=face_result.decisions_by_track,
            label_resolver=running_beer_label,
        )
        typer.secho(
            f"Face labels updated with running beer counts in {output_path}",
            fg=typer.colors.GREEN,
        )
    finally:
        if log_handle is not None:
            log_handle.close()


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
