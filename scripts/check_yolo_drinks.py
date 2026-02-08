#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


DRINK_LABELS = {"bottle", "cup", "wine glass"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check whether a YOLO model detects drink classes in a video frame."
    )
    parser.add_argument(
        "video",
        type=Path,
        help="Path to an input video (e.g. data/11-30-1.mp4).",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="yolov8n.pt",
        help="Path to YOLO weights (default: yolov8n.pt).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.15,
        help="YOLO confidence threshold (default: 0.15).",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="0-based frame index to inspect (default: 0).",
    )
    parser.add_argument(
        "--max-print",
        type=int,
        default=80,
        help="Maximum number of detections to print (default: 80).",
    )
    args = parser.parse_args()

    if not args.video.exists():
        raise SystemExit(f"Video not found: {args.video}")

    model = YOLO(args.model)
    names = model.model.names
    print(f"num_classes: {len(names)}")
    for label in sorted(DRINK_LABELS):
        print(f"has {label}: {any(v == label for v in names.values())}")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Unable to open video: {args.video}")
    if args.frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Unable to read frame {args.frame} from {args.video}")

    res = model(frame, conf=args.conf, verbose=False)[0]
    dets: list[tuple[str, float]] = []
    for b in res.boxes:
        cls = int(b.cls[0])
        conf = float(b.conf[0])
        name = res.names.get(cls, str(cls))
        dets.append((name, conf))
    dets.sort(key=lambda x: -x[1])

    people = sum(1 for name, _ in dets if name == "person")
    drinks = sum(1 for name, _ in dets if name in DRINK_LABELS)
    print(f"frame={args.frame} conf>={args.conf}: people={people} drinks={drinks}")

    printed = 0
    for name, conf in dets:
        if printed >= args.max_print:
            break
        if name == "person" or name in DRINK_LABELS:
            print(f"{name}\t{conf:.2f}")
            printed += 1


if __name__ == "__main__":
    main()

