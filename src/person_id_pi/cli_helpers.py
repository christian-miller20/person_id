from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import typer

from .beverage_config import BeverageDetectorConfig
from .beverage_detector import YoloBeverageDetector
from .beverage_pipeline import BeveragePipeline
from .beverage_store import BeverageStore
from .face_pipeline import FacePipeline, FrameTrackAnnotation
from .face_types import IdentityDecision
from .identity_engine import IdentityEngine


@dataclass(frozen=True)
class FaceStageResult:
    frame_annotations: list[list[FrameTrackAnnotation]]
    decisions_by_track: dict[int, IdentityDecision]


def run_face_stage(
    *, 
    source: str,
    identity: IdentityEngine,
    pipeline: FacePipeline,
    output_path: Path,
    limit_frames: Optional[int],
    verbose: bool,
    log_fn: Optional[Callable[[str], None]],
    update_templates: bool,
    auto_enroll_unknown: bool,
) -> FaceStageResult:
    tracklets_by_id, frame_annotations = (
        pipeline.extract_tracklets_with_annotations_from_video(
            source=source,
            limit_frames=limit_frames,
            verbose=verbose,
            log_fn=log_fn,
        )
    )
    decisions_by_track: dict[int, IdentityDecision] = {}
    for track_id in sorted(tracklets_by_id.keys()):
        tracklet = tracklets_by_id[track_id]
        decision = identity.match(tracklet)
        auto_enroll_status: Optional[str] = None
        if (
            update_templates
            and decision.user_id
            and identity.should_update_templates(decision, tracklet)
        ):
            identity.update_templates(decision.user_id, tracklet)
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
    return FaceStageResult(
        frame_annotations=frame_annotations,
        decisions_by_track=decisions_by_track,
    )


def run_beverage_stage(
    *,
    source: str,
    face_result: FaceStageResult,
    events_store: Path,
    count_beers: bool,
    count_espressos: bool,
    detector_model: Optional[str],
    object_conf_min: Optional[float],
    association_max_dist: Optional[float],
    limit_frames: Optional[int],
    verbose: bool,
    log_fn: Optional[Callable[[str], None]],
) -> None:
    track_to_user: dict[int, str] = {
        track_id: decision.user_id
        for track_id, decision in face_result.decisions_by_track.items()
        if decision.accepted and decision.user_id is not None
    }
    if not track_to_user:
        typer.secho(
            "No accepted identities; skipping beverage event extraction.",
            fg=typer.colors.YELLOW,
        )
        return

    detector_config = BeverageDetectorConfig(
        model_path=detector_model or BeverageDetectorConfig.model_path,
        conf_min=(
            object_conf_min
            if object_conf_min is not None
            else BeverageDetectorConfig.conf_min
        ),
    )
    detector = YoloBeverageDetector(config=detector_config)
    beverage_pipeline = BeveragePipeline(
        detector=detector,
        association_max_dist_norm=(
            association_max_dist if association_max_dist is not None else 0.25
        ),
    )
    events = beverage_pipeline.build_events(
        source=source,
        frame_annotations=face_result.frame_annotations,
        track_to_user=track_to_user,
        count_beers=count_beers,
        count_espressos=count_espressos,
        limit_frames=limit_frames,
        verbose=verbose,
        log_fn=log_fn,
    )

    store = BeverageStore(events_store)
    added = 0
    for event in events:
        if store.add_event(event):
            added += 1

    beer_totals = store.total_beers_by_user()
    espresso_totals = store.total_espressos_by_user()
    typer.secho(
        f"Beverage events: detected={len(events)} persisted_new={added} store={events_store}",
        fg=typer.colors.GREEN,
    )
    for user_id in sorted(set(beer_totals.keys()) | set(espresso_totals.keys())):
        typer.secho(
            f"user={user_id} beers_total={beer_totals.get(user_id, 0)} "
            f"espressos_total={espresso_totals.get(user_id, 0)}",
            fg=typer.colors.BLUE,
        )
