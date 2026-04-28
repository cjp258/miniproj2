from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import cv2
import torch

from src.detrac_dataset import ID_TO_CLASS
from src.modeling import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--video-path", type=str, required=True)
    p.add_argument("--output-video", type=str, required=True)
    p.add_argument("--output-json", type=str, default="traffic_stats.json")
    p.add_argument("--score-thresh", type=float, default=0.5)
    p.add_argument("--window-seconds", type=int, default=60)
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=5)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video = Path(args.output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_idx = 0
    max_road_load = 0

    # Sliding window count proxy for traffic flow.
    window_size = max(1, int(args.window_seconds * fps))
    window_counts = deque()
    running_sum = 0
    max_traffic_flow_window = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x = torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        x = x.to(device)

        with torch.no_grad():
            pred = model(x)[0]

        boxes = pred["boxes"].detach().cpu().numpy()
        labels = pred["labels"].detach().cpu().numpy()
        scores = pred["scores"].detach().cpu().numpy()

        visible_count = 0
        for box, label, score in zip(boxes, labels, scores):
            if score < args.score_thresh:
                continue

            x1, y1, x2, y2 = map(int, box.tolist())
            cls_name = ID_TO_CLASS.get(int(label), "unknown")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{cls_name} {score:.2f}",
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )
            visible_count += 1

        max_road_load = max(max_road_load, visible_count)

        window_counts.append(visible_count)
        running_sum += visible_count
        if len(window_counts) > window_size:
            running_sum -= window_counts.popleft()
        max_traffic_flow_window = max(max_traffic_flow_window, running_sum)

        cv2.putText(
            frame,
            f"Frame: {frame_idx} | Visible: {visible_count} | MaxLoad: {max_road_load}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    stats = {
        "video_path": args.video_path,
        "fps": fps,
        "frames_processed": frame_idx,
        "window_seconds": args.window_seconds,
        "max_road_load": int(max_road_load),
        "max_traffic_flow_window": int(max_traffic_flow_window),
        "note": "Flow is a baseline sliding-window count of detections; tracking can improve uniqueness.",
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))
    print(f"Annotated video saved to: {output_video}")


if __name__ == "__main__":
    main()
