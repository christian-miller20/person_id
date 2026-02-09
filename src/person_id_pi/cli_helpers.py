from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, TextIO

import typer

from .beverage_config import BeverageDetectorConfig
from .beverage_detector import MultiBeverageDetector, YoloBeverageDetector
from .beverage_pipeline import BeveragePipeline
from .beverage_store import BeverageStore
from .beverage_types import BeverageEvent
from .face_pipeline import FacePipeline, FrameTrackAnnotation
from .face_types import IdentityDecision
from .identity_engine import IdentityEngine


@dataclass(frozen=True)
class FaceStageResult:
    frame_annotations: list[list[FrameTrackAnnotation]]
    decisions_by_track: dict[int, IdentityDecision]


@dataclass(frozen=True)
class BeverageStageResult:
    beer_totals: dict[str, int]
    espresso_totals: dict[str, int]
    detected_events: int
    persisted_new: int
    events: list[BeverageEvent]
    persisted_events: list[BeverageEvent]


def default_annotate_output_path(source: str) -> Path:
    src = Path(source)
    stem = src.stem if src.suffix else src.name
    return Path("runs") / f"{stem}_annotated.mp4"


def default_log_path(source: str) -> Path:
    src = Path(source)
    stem = src.stem if src.suffix else src.name
    return Path("logs") / f"{stem}.log"


def build_log_writer(
    source: str,
    verbose: bool,
    tee_logs: bool,
) -> tuple[Optional[Callable[[str], None]], Optional[TextIO]]:
    if not verbose:
        return None, None
    log_path = default_log_path(source)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")
    typer.secho(f"Verbose logs written to {log_path}", fg=typer.colors.BLUE)

    def _file_only(message: str) -> None:
        log_handle.write(f"{message}\n")

    def _tee(message: str) -> None:
        log_handle.write(f"{message}\n")
        print(message)

    return (_tee if tee_logs else _file_only), log_handle


def build_running_beer_label_resolver(
    events: list[BeverageEvent],
    baseline_beers_by_user: Optional[dict[str, int]] = None,
) -> Callable[[int, int, Optional[IdentityDecision]], Optional[str]]:
    beer_events_by_frame: dict[int, list[str]] = defaultdict(list)
    for event in events:
        if event.beverage_label in {"cup", "can", "bottle"}:
            beer_events_by_frame[event.frame_idx].append(event.user_id)

    running_beers_by_user: dict[str, int] = defaultdict(int)
    if baseline_beers_by_user:
        for user_id, total in baseline_beers_by_user.items():
            running_beers_by_user[user_id] = int(total)
    last_applied_frame = -1

    def _resolver(
        frame_idx: int, track_id: int, decision: Optional[IdentityDecision]
    ) -> Optional[str]:
        nonlocal last_applied_frame
        _ = track_id
        if frame_idx > last_applied_frame:
            for idx in range(last_applied_frame + 1, frame_idx + 1):
                for user_id in beer_events_by_frame.get(idx, []):
                    running_beers_by_user[user_id] += 1
            last_applied_frame = frame_idx
        if not (decision and decision.accepted and decision.user_id):
            return None
        return (
            f"{decision.user_id} beers={running_beers_by_user.get(decision.user_id, 0)}"
        )

    return _resolver


def compute_baseline_beers_by_user(
    *,
    beer_totals: dict[str, int],
    persisted_events: list[BeverageEvent],
) -> dict[str, int]:
    """
    Compute lifetime beer totals *before* this run's newly persisted events.

    This keeps running overlay labels stable and prevents double-attributing
    counts when we replay current-video events on the timeline.
    """
    this_run_persisted_beers: dict[str, int] = {}
    for event in persisted_events:
        if event.beverage_label not in {"cup", "can", "bottle"}:
            continue
        this_run_persisted_beers[event.user_id] = (
            this_run_persisted_beers.get(event.user_id, 0) + 1
        )
    return {
        user_id: max(0, total - this_run_persisted_beers.get(user_id, 0))
        for user_id, total in beer_totals.items()
    }


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
    annotate_output_path: Optional[Path],
    face_result: FaceStageResult,
    events_store: Path,
    count_beers: bool,
    count_espressos: bool,
    detector_model: Optional[str],
    object_conf_min: Optional[float],
    secondary_detector_model: Optional[str],
    secondary_object_conf_min: Optional[float],
    secondary_allowed_labels: Optional[str],
    secondary_merge_iou: float,
    association_max_dist: Optional[float],
    limit_frames: Optional[int],
    beverage_hold_frames: int,
    verbose: bool,
    log_fn: Optional[Callable[[str], None]],
) -> BeverageStageResult:
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
        return BeverageStageResult(
            beer_totals={},
            espresso_totals={},
            detected_events=0,
            persisted_new=0,
            events=[],
            persisted_events=[],
        )

    detector_config = BeverageDetectorConfig(
        model_path=detector_model or BeverageDetectorConfig.model_path,
        conf_min=(
            object_conf_min
            if object_conf_min is not None
            else BeverageDetectorConfig.conf_min
        ),
    )
    detector = YoloBeverageDetector(config=detector_config)
    if secondary_detector_model:
        labels_raw = secondary_allowed_labels or "espresso_shot"
        labels = tuple(
            label.strip() for label in labels_raw.split(",") if label.strip()
        )
        secondary_config = BeverageDetectorConfig(
            model_path=secondary_detector_model,
            conf_min=(
                secondary_object_conf_min
                if secondary_object_conf_min is not None
                else BeverageDetectorConfig.conf_min
            ),
            allowed_labels=labels or ("espresso_shot",),
        )
        secondary_detector = YoloBeverageDetector(config=secondary_config)
        detector = MultiBeverageDetector(
            [detector, secondary_detector],
            dedupe_iou_threshold=secondary_merge_iou,
        )
        typer.secho(
            "Using dual beverage detectors (primary + secondary).",
            fg=typer.colors.BLUE,
        )
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
    persisted_events: list[BeverageEvent] = []
    for event in events:
        if store.add_event(event):
            added += 1
            persisted_events.append(event)

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

    if annotate_output_path is not None:
        beverage_pipeline.annotate_video_in_place(
            video_path=annotate_output_path,
            count_beers=count_beers,
            count_espressos=count_espressos,
            limit_frames=limit_frames,
            hold_frames=beverage_hold_frames,
            verbose=verbose,
            log_fn=log_fn,
        )
        typer.secho(
            f"Beverage boxes written to {annotate_output_path}",
            fg=typer.colors.GREEN,
        )
    return BeverageStageResult(
        beer_totals=beer_totals,
        espresso_totals=espresso_totals,
        detected_events=len(events),
        persisted_new=added,
        events=events,
        persisted_events=persisted_events,
    )
